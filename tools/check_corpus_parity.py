#!/usr/bin/env python3
"""Check complete CPU and accelerator grad-corpus bundles and the dispatch log."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from w14b_acceptance import placement


def main():
    ap = argparse.ArgumentParser()
    for k in ("cpu", "gpu", "spec", "corpus", "log", "report"):
        ap.add_argument("--" + k, type=Path, required=True)
    a = ap.parse_args()
    spec = json.loads(a.spec.read_bytes())
    cpu, gpu = [json.loads((p/"gradient.json").read_bytes()) for p in (a.cpu, a.gpu)]
    assert cpu["backend"] == "cpu" and gpu["backend"] in {"cuda", "rocm"}
    assert cpu["tokens"] == gpu["tokens"] and len(cpu["tokens"]) == len(a.corpus.read_text().splitlines())
    for m in (cpu, gpu):
        assert m["schema_version"] == 1 and m["objective"] == "mean_next_token_nll"
        assert m["corpus_sha256"] == hashlib.sha256(a.corpus.read_bytes()).hexdigest() == spec["corpus_sha256"]
        assert m["observation_count"] == sum(len(s)-1 for s in m["tokens"])
        assert m["dtype"] == "f32_le" and m["placement"]["strict"] is True
        assert [r["name"] for r in m["gradients"]] == spec["weights"]
    audit = placement(a.log.read_text(), gpu["placement"]["device"], {"backward"})
    assert audit["backward"]["graphs"] == sum(len(s) > 1 for s in cpu["tokens"])
    rows = []
    for x, y in zip(cpu["gradients"], gpu["gradients"], strict=True):
        assert x["shape"] == y["shape"]
        data = []
        for p, r in [(a.cpu, x), (a.gpu, y)]:
            assert Path(r["file"]).name == r["file"]
            path = p / r["file"]
            with path.open("rb") as f:
                assert hashlib.file_digest(f, "sha256").hexdigest() == r["sha256"]
            assert path.stat().st_size == 4*math.prod(x["shape"])
            data.append(np.memmap(path, dtype="<f4", mode="r"))
        peak = err = 0.
        for i in range(0, len(data[0]), 1 << 19):
            u, v = [t[i:i+(1 << 19)] for t in data]
            assert np.isfinite(u).all() and np.isfinite(v).all()
            peak = max(peak, float(np.max(abs(u))))
            err = max(err, float(np.max(abs(u-v))))
        rows.append(dict(name=x["name"], max_abs=err, reference_peak=peak,
                         relative_linf=err/peak if peak else (math.inf if err else 0.)))
    report = dict(backend=gpu["backend"], placement=audit, loss_error=abs(cpu["loss"]-gpu["loss"]),
                  observation_count=cpu["observation_count"], rows=rows)
    a.report.write_text(json.dumps(report, indent=2))
    worst = max(r["relative_linf"] for r in rows)
    print(f"maximum relative Linf: {worst}, loss error: {report['loss_error']}, {audit}")
    assert worst <= .001 and report["loss_error"] <= 5e-5


if __name__ == "__main__":
    main()
