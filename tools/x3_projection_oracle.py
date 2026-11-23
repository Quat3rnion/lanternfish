#!/usr/bin/env python3
"""Build a test-only observer and verify the W14b final projection with FP64.

Build inside a build slot. Run compare inside a GPU turn, with llama.cpp's
gguf-py on PYTHONPATH. Neither action changes the production build or pin.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

import numpy as np

from w14b_acceptance import MODEL_HASH, digest, placement


def build(args):
    work = args.witness.resolve()
    work.mkdir(parents=True, exist_ok=True)
    builddir = args.llama / "build"
    source = args.llama / "src/llama-context.cpp"
    text = source.read_text()
    a = "    if (!ggml_backend_sched_alloc_graph(sched.get(), graph)) {"
    b = "    ggml_backend_tensor_get(grad, grad_out, 0, ggml_nbytes(grad));"
    assert text.count(a) == text.count(b) == 1
    text = text.replace(a, """
    auto * d_logits = ggml_graph_get_grad(graph, output);
    auto * d_hidden = ggml_graph_get_grad(graph, output->src[1]);
    if (!d_logits || !d_hidden) throw std::runtime_error("missing projection adjoints");
    ggml_set_output(d_logits);
    ggml_set_output(d_hidden);
""" + a).replace(b, """
    for (auto p : {std::make_pair(d_logits,"g_logits"), std::make_pair(d_hidden,"g_hidden")}) {
        std::vector<float> v(ggml_nelements(p.first));
        ggml_backend_tensor_get(p.first, v.data(), 0, v.size()*sizeof(float));
        FILE * f = fopen(p.second, "wb");
        if (!f || fwrite(v.data(),sizeof(float),v.size(),f) != v.size() || fclose(f))
            throw std::runtime_error("could not save oracle operand");
    }
""" + b)
    witness = work/source.name
    witness.write_text(text)
    target = next((builddir/"bin").glob("libllama.so.0.*.*")).relative_to(builddir)
    cmds = subprocess.check_output(["ninja", "-C", builddir, "-t", "commands", target], text=True).splitlines()
    cc = shlex.split(next(x for x in cmds if x.endswith(" -c " + str(source))))
    obj = cc[cc.index("-o")+1]
    cc = [str(work/"ctx.o") if x == obj else str(work/"ctx.d") if x == obj+".d"
          else str(witness) if x == str(source) else x for x in cc]
    subprocess.run(cc, cwd=builddir, check=True)
    ld = shlex.split(cmds[-1].removeprefix(": && ").removesuffix(" && :"))
    ld = [str(work/"ctx.o") if x == obj else str(work/"libllama.so.0") if x == str(target)
          else "-Wl,--dependency-file="+str(work/"link.d") if x.startswith("-Wl,--dependency-file=")
          else x for x in ld]
    subprocess.run(ld, cwd=builddir, check=True)


def compare(args):
    from gguf import GGUFReader
    pin, work, model = args.pin.resolve(), args.witness.resolve(), args.model.resolve()
    assert digest(model) == MODEL_HASH
    weight = next(t.data for t in GGUFReader(str(model)).tensors if t.name == "token_embd.weight")
    assert weight.shape == (151936, 896) and weight.dtype == np.float32
    spec = dict(tokens=[26567, 362, 20, 572, 3465, 553], target_id=5085,
                site=dict(kind="weight", name="blk.23.ffn_down.weight"))
    if args.objective == "target_logit":
        spec["objective"] = args.objective
    report = dict(spec=spec, model_sha256=MODEL_HASH, binary_sha256=digest(pin/"lanternfish"),
                  witness_sha256=digest(work/"libllama.so.0"), cases={})
    for backend in ("cpu", "cuda", "rocm"):
        libs = pin/"lib"/backend
        result = dict(libraries={p.name:digest(p) for p in libs.glob("lib*.so.*")})
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            (cwd/"spec").write_text(json.dumps(spec))
            for mode in ("production", "observer"):
                ld = str(libs) if mode == "production" else f"{work}:{libs}"
                env = dict(os.environ, LD_LIBRARY_PATH=ld)
                out = cwd/mode
                cmd = [pin/"lanternfish", "--verbose", "grad-select", "--model", model,
                       "--spec", cwd/"spec", "--out", out, "--backend", backend, "--threads", "6"]
                run = subprocess.run(cmd, env=env, cwd=cwd, text=True, capture_output=True)
                if run.returncode:
                    raise RuntimeError(run.stderr[-4000:])
                meta = json.loads((out/"gradient.json").read_text())
                assert meta["sha256"] == digest(out/"gradient.f32") == result.setdefault("gradient_sha256", meta["sha256"])
                if backend != "cpu":
                    placement(run.stderr, meta["placement"]["device"], {"backward"})
            g, h = (np.fromfile(cwd/name, "<f4").astype("d") for name in ("g_logits", "g_hidden"))
            assert g.shape == (151936,) and h.shape == (896,) and np.isfinite(g).all() and np.isfinite(h).all()
            if args.objective == "target_logit":
                assert g[spec["target_id"]] == 1. and np.count_nonzero(g) == 1
            ref = np.zeros(896)
            for i in range(0, len(g), 1024):
                ref += g[i:i+1024] @ weight[i:i+1024].astype("d")
            result.update(max_abs_error=float(np.max(abs(h-ref))), g_hidden_sha256=digest(cwd/"g_hidden"),
                          g_logits_sha256=digest(cwd/"g_logits"), fp64_sha256=hashlib.sha256(ref.astype("<f8")).hexdigest())
        report["cases"][backend] = result
        args.report.write_text(json.dumps(report, indent=2))
        print(backend, result["max_abs_error"], flush=True)
    limit = min(report["cases"]["cpu"]["max_abs_error"], 1e-5)
    assert all(report["cases"][b]["max_abs_error"] <= limit for b in ("cuda", "rocm"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    subs = ap.add_subparsers(required=True)
    for fn, keys in [(build, ("witness", "llama")), (compare, ("witness", "pin", "model", "report"))]:
        p = subs.add_parser(fn.__name__)
        for key in keys:
            p.add_argument("--"+key, type=Path, required=True)
        if fn is compare:
            p.add_argument("--objective", choices=("target_logit", "next_token_cross_entropy"),
                           default="next_token_cross_entropy")
        p.set_defaults(action=fn)
    parsed = ap.parse_args()
    parsed.action(parsed)
