"""Native locate-then-edit CPU regression on independently generated GGUFs."""
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from tiny_gguf import write_model


class EditTests(unittest.TestCase):
    def test_rome_optimizes_teacher_forced_target_on_dense_models(self):
        for arch, dtype in (("llama", "f32"), ("qwen2", "bf16")):
            with self.subTest(arch=arch, dtype=dtype), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                model = root / "tiny.gguf"
                request = root / "request.json"
                result = root / "result.json"
                write_model(model, dtype, arch)
                parent_bytes = model.read_bytes()
                request.write_text(json.dumps({
                    "method": "rome", "layers": [0],
                    "facts": [{"subject": "abc", "prompt": "{}d", "target": " ef"}],
                    "steps": 5,
                }))
                job = subprocess.run([
                    os.environ["LANTERNFISH_BIN"], "locate-edit", "--model", model,
                    "--request", request, "--out", result, "--cache", root / "cache",
                    "--threads", "2", "--verbose",
                ], capture_output=True, text=True)
                self.assertEqual(job.returncode, 0, job.stderr)
                manifest = json.loads(result.read_text())
                self.assertEqual(manifest["source_dtype"], dtype)
                self.assertEqual(manifest["gradient_dtype"], "f32")
                self.assertLess(manifest["losses"][0][1], manifest["losses"][0][0])
                update, = manifest["updates"]
                self.assertEqual(update["layer"], 0)
                direction, = update["keys"]
                residual, = update["values"]
                self.assertEqual((len(direction), len(residual)), (64, 32))
                self.assertTrue(all(map(math.isfinite, direction + residual)))
                self.assertEqual(model.read_bytes(), parent_bytes)

    def test_two_layer_batch_uses_pinned_moments_and_f32_backward(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "tiny.gguf"
            request = root / "request.json"
            cache = root / "cache"
            write_model(model, "bf16", "qwen2")
            parent = model.read_bytes()
            for method in ("memit", "alphaedit"):
                with self.subTest(method=method):
                    out = root / f"{method}.json"
                    request.write_text(json.dumps({
                        "method": method, "layers": [0, 1], "steps": 3,
                        "facts": [{"subject": s, "prompt": "d{}", "target": t}
                                  for s, t in (("abc", " ef"), ("bc", " ag"))],
                    }))
                    job = subprocess.run([
                        os.environ["LANTERNFISH_BIN"], "locate-edit", "--model", model,
                        "--request", request, "--out", out, "--cache", cache,
                        "--threads", "2", "--verbose",
                    ], capture_output=True, text=True)
                    self.assertEqual(job.returncode, 0, job.stderr)
                    result = json.loads(out.read_text())
                    self.assertEqual(result["source_dtype"], "bf16")
                    self.assertEqual([u["layer"] for u in result["updates"]], [0, 1])
                    for u in result["updates"]:
                        self.assertEqual([len(v) for v in u["keys"]], [64] * 4)
                        self.assertEqual([len(v) for v in u["values"]], [32] * 4)
                    self.assertTrue(all(a > b for a, b in result["losses"]))
                    current = {p.name: p.stat().st_mtime_ns for p in cache.iterdir()}
                    self.assertEqual(len(current), 2)
                    if method == "memit":
                        stamps = current
                    else:
                        self.assertEqual(current, stamps)
            self.assertEqual(model.read_bytes(), parent)
