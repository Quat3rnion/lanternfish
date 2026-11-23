#!/usr/bin/env python3
"""Compare grad-select on two held-out W14b cases. Run inside the GPU queue."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from w14b_acceptance import MODEL_HASH, placement


def main():
    ap = argparse.ArgumentParser()
    for key in ("pin", "model", "evidence"):
        ap.add_argument("--" + key, type=Path, required=True)
    ap.add_argument("--backends", nargs="+", default=["cpu", "cuda", "rocm"])
    ap.add_argument("--objective", choices=("next_token_cross_entropy", "target_logit"),
                    default="next_token_cross_entropy")
    args = ap.parse_args()
    assert args.backends[0] == "cpu"
    args.evidence.mkdir(exist_ok=True, parents=True)
    with args.model.open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == MODEL_HASH
    cases = [("paris", [785, 6722, 315, 9625, 374], 12095),
             ("zero", [28253, 93377, 518], 7168)]
    sites = [dict(kind="weight", name=name) for name in
             ("blk.0.attn_q.weight", "blk.12.attn_v.bias", "blk.23.ffn_down.weight")]
    sites += [dict(kind="activation", layer=layer, position=position, role=role)
              for role, layer, position in [("l_out", 0, 0), ("attn_out", 12, 1), ("ffn_out", 23, -1)]]
    report = []
    for case, tokens, target in cases:
        for site in sites:
            if site.get("position") == -1:
                site = dict(site, position=len(tokens)-1)
            spec = dict(tokens=tokens, target_id=target, site=site)
            if args.objective == "target_logit":
                spec["objective"] = args.objective
            with tempfile.TemporaryDirectory() as w:
                work = Path(w)
                (work/"spec").write_text(json.dumps(spec))
                for backend in args.backends:
                    out = work/backend
                    libs = args.pin/"lib"/backend
                    env = dict(os.environ, LD_LIBRARY_PATH=str(libs))
                    argsv = [args.pin/"lanternfish", "--verbose", "grad-select", "--spec", work/"spec",
                             "--out", out, "--threads", "6", "--model", args.model, "--backend", backend]
                    result = subprocess.run(argsv, env=env, capture_output=True, text=True)
                    if result.returncode:
                        raise RuntimeError(f"{case} {site} {backend}: {result.stderr[-3000:]}")
                    meta = json.loads((out/"gradient.json").read_text())
                    data = np.fromfile(out/"gradient.f32", dtype="<f4")
                    assert meta["site"] == site and meta["backend"] == backend and np.isfinite(data).all()
                    assert meta["objective"] == args.objective
                    scalar = meta["logit" if args.objective == "target_logit" else "loss"]
                    if backend == "cpu":
                        ref, ref_scalar = data, scalar
                    err = float(np.max(np.abs(ref-data)))
                    peak = float(np.max(np.abs(ref)))
                    rel = err/peak if peak else (0 if err == 0 else float("inf"))
                    row = dict(case=case, spec=spec, **meta, relative_linf=rel, scalar_error=abs(scalar-ref_scalar))
                    if backend != "cpu":
                        row["audit"] = placement(result.stderr, meta["placement"]["device"], {"backward"})
                    report.append(row)
                    (args.evidence/"parity.json").write_text(json.dumps(report, indent=2))
                    print(case, site, backend, rel, row["scalar_error"], flush=True)
    assert all(row["relative_linf"] <= .001 and row["scalar_error"] <= 5e-5 for row in report)


if __name__ == "__main__":
    main()
