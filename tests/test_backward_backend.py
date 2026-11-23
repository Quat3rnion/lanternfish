"""Backend selection at the grad and locate-edit CLI boundaries."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from tiny_gguf import write_model


class BackwardBackendTests(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.root = Path(work.name)
        self.model = self.root / "tiny.gguf"
        write_model(self.model, "f32")
        self.request = self.root / "request.json"
        self.request.write_text(json.dumps({
            "method": "rome", "layers": [0], "steps": 1,
            "facts": [{"subject": "abc", "prompt": "{}d", "target": " ef"}],
        }))

    def run_backward(self, cmd, out, *args, env=None):
        options = {
            "grad": ["--prompt", "abc", "--layer", "0", "--target-token", "6"],
            "locate-edit": ["--request", self.request, "--cache", self.root / "cache"],
        }
        return subprocess.run([
            os.environ["LANTERNFISH_BIN"], cmd, "--model", self.model,
            "--out", out, "--threads", "2", *options[cmd], *args,
        ], capture_output=True, text=True, env=env, timeout=30)

    def test_default_and_explicit_cpu_record_backend_and_device(self):
        for cmd in ("grad", "locate-edit"):
            for args in ((), ("--backend", "cpu")):
                with self.subTest(cmd=cmd, args=args):
                    out = self.root / f"{cmd}-{len(args)}.json"
                    job = self.run_backward(cmd, out, *args)
                    self.assertEqual(job.returncode, 0, job.stderr)
                    manifest = out / "manifest.json" if cmd == "grad" else out
                    provenance = json.loads(manifest.read_text())["provenance"]
                    self.assertEqual(provenance["backend"], "cpu")
                    self.assertEqual(provenance["device"], "CPU")

    def test_cuda_without_visible_device_fails_without_export(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", NVIDIA_TF32_OVERRIDE="1",
                   GGML_CUDA_CUBLAS_COMPUTE_TYPE="f16")
        for cmd in ("grad", "locate-edit"):
            with self.subTest(cmd=cmd):
                out = self.root / f"{cmd}.json"
                job = self.run_backward(cmd, out, "--backend", "cuda", env=env)
                self.assertGreater(job.returncode, 0, job.stderr)
                self.assertIn("cuda backend unavailable", job.stderr)
                self.assertFalse(out.exists())
