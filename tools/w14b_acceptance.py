#!/usr/bin/env python3
"""Check explicit CPU/CUDA gradients and time the W9 ROME factor calculation.

Run both backends under gpu-turn.sh (with build-gate.sh quiet for timing).
Pass --cpu-only to check the CPU reference without a GPU turn.
"""

import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from tiny_gguf import write_model

MODEL = Path("/arboretum/w14b/qwen-f32.gguf")
MODEL_HASH = "8df5d93dbefb5f73ce905c6e464aaa312756fab7a8a85ee1f7dae15cec36bfde"
GRAD_ATOL = 1e-5
GRAD_RTOL = 2e-3
LOSS_ATOL = 5e-5
FD_ATOL = 3e-4
FD_STEP = 0.003
QWEN_LAYERS = (0, 12, 23)
ROME_REQUEST = {
    "method": "rome", "layers": [12], "steps": 16,
    "facts": [{"subject": "Apple A5", "prompt": "{} was created by", "target": "Google"}],
}
QWEN_PROMPT = "Apple A5 was created by"
QWEN_TARGET = 5085  # Qwen2.5 token ID for " Google".


def compare(reference, actual, *, atol, rtol):
    """Check each element against atol + rtol*abs(reference)."""
    if not reference or len(reference) != len(actual):
        raise ValueError("gradient vectors must have the same nonzero length")
    max_abs = max_ref = max_ratio = 0.0
    for ref, val in zip(reference, actual, strict=True):
        if not (math.isfinite(ref) and math.isfinite(val)):
            raise ValueError("non-finite comparison input")
        diff = abs(val - ref)
        max_abs = max(max_abs, diff)
        max_ref = max(max_ref, abs(ref))
        max_ratio = max(max_ratio, diff / (atol + rtol * abs(ref)))
    return {
        "elements": len(reference),
        "max_absolute_error": max_abs,
        "relative_linf_error": max_abs / max_ref if max_ref else None,
        "max_tolerance_ratio": max_ratio,
        "passed": max_ratio <= 1,
    }


def cross_entropy(top, target, vocab_size):
    logits = {row["token_id"]: row["logit"] for row in top}
    if (len(top) != vocab_size or logits.keys() != set(range(vocab_size))
            or not all(map(math.isfinite, logits.values()))):
        raise ValueError("expected one finite logit for every vocabulary token")
    peak = max(logits.values())
    return math.log(math.fsum(math.exp(v - peak) for v in logits.values())) + (peak - logits[target])


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(path.read_text())


def check_backend(doc, backend):
    provenance = doc["provenance"]
    require(provenance["backend"] == backend and provenance["device"],
            f"incorrect {backend} provenance: {provenance}")
    return provenance


def run_cli(args, label, command, *options, audit=False):
    env = dict(os.environ)
    env.pop("GGML_SCHED_DEBUG", None)
    if audit:
        env["GGML_SCHED_DEBUG"] = "2"
    if args.cpu_only:
        env["CUDA_VISIBLE_DEVICES"] = ""
    log = args.evidence / f"{label}.log"
    cmd = [args.binary, command, *options, "--threads", str(args.threads)]
    if audit:
        cmd += ["--verbose"]
    with log.open("w") as out:
        start = time.perf_counter()
        result = subprocess.run(cmd, env=env, stdout=out, stderr=subprocess.STDOUT,
                                timeout=600, check=False)
        seconds = time.perf_counter() - start
    require(result.returncode == 0, f"{label}: exit {result.returncode}; see {log}")
    return seconds


def gradient(args, work, label, model, prompt, target, layer, shape, backend):
    out = work / label
    run_cli(args, label, "grad", "--model", model, "--prompt", prompt,
            "--target-token", str(target), "--layer", str(layer), "--out", out,
            "--backend", backend, audit=(backend == "cuda"))
    manifest = out / "manifest.json"
    doc = load_json(manifest)
    (args.evidence / f"{label}-manifest.json").write_bytes(manifest.read_bytes())
    weight = f"blk.{layer}.ffn_down.weight"
    require(doc["kind"] == "gradient" and doc["weight"] == weight
            and doc["layer"] == layer and doc["target_token"] == target
            and doc["objective"] == "next_token_cross_entropy"
            and doc["position"] == len(doc["tokens"]) - 1 and math.isfinite(doc["loss"]),
            f"{label}: invalid gradient manifest")
    check_backend(doc, backend)
    tensor = doc["tensor"]
    tensor_file = out / tensor["file"]
    payload = tensor_file.read_bytes()
    require(tensor["dtype"] == "f32_le" and tensor["shape"] == shape
            and len(payload) == tensor["bytes"] == 4 * math.prod(shape)
            and hashlib.sha256(payload).hexdigest() == tensor["sha256"],
            f"{label}: invalid gradient payload")
    grad = array("f")
    grad.frombytes(payload)
    if sys.byteorder != "little":
        grad.byteswap()
    require(all(map(math.isfinite, grad)), f"{label}: non-finite gradient")
    tensor_file.unlink()
    return doc, grad


def compare_gradients(cpu, cuda):
    cpu_doc, cpu_grad = cpu
    cuda_doc, cuda_grad = cuda
    require(cpu_doc["tokens"] == cuda_doc["tokens"], "gradient tokenization mismatch")
    stats = compare(cpu_grad, cuda_grad, atol=GRAD_ATOL, rtol=GRAD_RTOL)
    loss_error = abs(cpu_doc["loss"] - cuda_doc["loss"])
    return {**stats, "cpu_loss": cpu_doc["loss"], "cuda_loss": cuda_doc["loss"],
            "absolute_loss_error": loss_error,
            "passed": stats["passed"] and loss_error <= LOSS_ATOL}


def finite_differences(args, work, layer, grads):
    cpu_doc, cpu_grad = grads["cpu"]
    prompt, target, vocab_size = "abc", cpu_doc["target_token"], 263
    spec = work / "fd-spec.json"
    spec.write_text(json.dumps({"layers": [], "roles": [], "positions": [cpu_doc["position"]],
                                "top_k": vocab_size, "byte_cap": 65536}))
    model = work / "fd.gguf"

    def trace_loss(label, perturb=None):
        write_model(model, "f32", "qwen2", perturb)
        out = work / label
        run_cli(args, label, "trace", "--model", model, "--prompt", prompt,
                "--spec", spec, "--out", out)
        doc = load_json(out / "manifest.json")
        require([t["token_id"] for t in doc["tokens"]] == cpu_doc["tokens"],
                f"{label}: trace tokenization mismatch")
        logits, = doc["model_logits"]
        return cross_entropy(logits["top"], target, vocab_size)

    baseline = trace_loss(f"fd-{layer}-base")
    # Check the strongest coordinate in each half of the weight, not just one op output.
    midpoint = len(cpu_grad) // 2
    rows = []
    for low, high in ((0, midpoint), (midpoint, len(cpu_grad))):
        index = max(range(low, high), key=lambda i: abs(cpu_grad[i]))
        require(abs(cpu_grad[index]) > 1e-3, f"fd-{layer}: gradient signal too small")
        weight = cpu_doc["weight"]
        plus = trace_loss(f"fd-{layer}-{index}-plus", (weight, index, FD_STEP))
        minus = trace_loss(f"fd-{layer}-{index}-minus", (weight, index, -FD_STEP))
        estimate = (plus - minus) / (2 * FD_STEP)
        for backend, (doc, grad) in grads.items():
            derivative_error = abs(grad[index] - estimate)
            loss_error = abs(doc["loss"] - baseline)
            rows.append({
                "backend": backend, "index": index, "analytical": grad[index],
                "centered_difference": estimate, "absolute_derivative_error": derivative_error,
                "absolute_loss_error": loss_error,
                "passed": derivative_error <= FD_ATOL and loss_error <= LOSS_ATOL,
            })
    return rows


def time_edit(args, work, backend, repeat):
    request = args.evidence / "rome-request.json"
    request.write_text(json.dumps(ROME_REQUEST, indent=2) + "\n")
    label = f"rome-{backend}-{repeat}"
    out = args.evidence / f"{label}.json"
    seconds = run_cli(args, label, "locate-edit", "--model", MODEL,
                      "--request", request, "--out", out,
                      "--cache", work / f"cache-{label}", "--backend", backend)
    doc = load_json(out)
    provenance = check_backend(doc, backend)
    require(doc["method"] == "rome" and doc["model_sha256"] == MODEL_HASH
            and doc["source_dtype"] == doc["gradient_dtype"] == "f32",
            f"ROME {backend}: incorrect edit provenance")
    update, = doc["updates"]
    key, = update["keys"]
    value, = update["values"]
    require(update["layer"] == 12 and (len(key), len(value)) == (4864, 896)
            and all(map(math.isfinite, key + value)) and any(key) and any(value),
            f"ROME {backend}: invalid rank-one factors")
    (before, after), = doc["losses"]
    return {"seconds": seconds, "provenance": provenance, "nll_before": before,
            "nll_after": after, "output_sha256": digest(out),
            "passed": math.isfinite(before) and math.isfinite(after) and after < before}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True, help="A new evidence directory")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--cpu-only", action="store_true", help="Check the CPU reference only")
    args = parser.parse_args()
    args.binary = args.binary.resolve(strict=True)
    args.evidence.mkdir(parents=True, exist_ok=False)
    backends = ("cpu",) if args.cpu_only else ("cpu", "cuda")
    report = {
        "qualification": "not release-qualified",
        "backends": backends,
        "binary": str(args.binary), "binary_sha256": digest(args.binary),
        "threads": args.threads, "model": str(MODEL), "model_sha256": digest(MODEL),
        "inherited_cuda_precision": {k: os.environ.get(k) for k in
                                     ("NVIDIA_TF32_OVERRIDE", "GGML_CUDA_CUBLAS_COMPUTE_TYPE")},
        "tolerance": {"gradient_absolute": GRAD_ATOL, "gradient_relative": GRAD_RTOL,
                      "loss_absolute": LOSS_ATOL, "finite_difference_absolute": FD_ATOL,
                      "finite_difference_step": FD_STEP},
        "qwen_prompt": QWEN_PROMPT, "qwen_target_token": QWEN_TARGET,
        "gradients": {}, "finite_differences": {}, "rome": {}, "passed": False,
    }
    checks = []
    try:
        require(report["model_sha256"] == MODEL_HASH, "Qwen F32 reference hash mismatch")
        with tempfile.TemporaryDirectory(prefix="w14b-", dir="/arboretum/tmp") as directory:
            work = Path(directory)
            tiny = work / "tiny.gguf"
            write_model(tiny, "f32", "qwen2")
            report["tiny_model_sha256"] = digest(tiny)
            suites = (("tiny", tiny, "abc", 6, (0, 1), [32, 64]),
                      ("qwen", MODEL, QWEN_PROMPT, QWEN_TARGET, QWEN_LAYERS, [896, 4864]))
            for name, model, prompt, target, layers, shape in suites:
                for layer in layers:
                    label = f"{name}-{layer}"
                    grads = {
                        b: gradient(args, work, f"{label}-{b}", model, prompt, target, layer, shape, b)
                        for b in backends
                    }
                    row = {b: {"loss": doc["loss"], "tensor_sha256": doc["tensor"]["sha256"],
                               "provenance": doc["provenance"]} for b, (doc, _) in grads.items()}
                    if "cuda" in grads:
                        row["parity"] = compare_gradients(grads["cpu"], grads["cuda"])
                        checks.append(row["parity"]["passed"])
                    report["gradients"][label] = row
                    if name == "tiny":
                        fd = finite_differences(args, work, layer, grads)
                        report["finite_differences"][label] = fd
                        checks.extend(row["passed"] for row in fd)
            samples = {b: [] for b in backends}
            for repeat in range(4):
                # Repeat 0 warms the file/system caches. Each invocation starts
                # with an empty moment cache and runs the complete factor command.
                order = backends if repeat % 2 == 0 else backends[::-1]
                for backend in order:
                    edit = time_edit(args, work, backend, repeat)
                    samples[backend].append(edit)
                    checks.append(edit["passed"])
            for backend, runs in samples.items():
                measured = [run["seconds"] for run in runs[1:]]
                report["rome"][backend] = {
                    "warmup": runs[0], "measured": runs[1:], "median_seconds": median(measured),
                    "min_seconds": min(measured), "max_seconds": max(measured),
                }
            if "cuda" in backends:
                report["rome"]["cpu_over_cuda_speedup"] = (
                    report["rome"]["cpu"]["median_seconds"] / report["rome"]["cuda"]["median_seconds"])
            require(digest(MODEL) == MODEL_HASH, "Qwen F32 reference changed during acceptance")
            report["passed"] = all(checks)
    except (KeyError, OSError, ValueError, subprocess.SubprocessError) as err:
        report["error"] = str(err)
    path = args.evidence / "report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
