"""CPU CLI regressions. Set LANTERNFISH_BIN and TMPDIR before running."""
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

from tiny_gguf import write_model


class TraceTests(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.work = Path(work.name)
        self.binary = os.environ["LANTERNFISH_BIN"]
        self.spec = {
            "layers": [0, 1], "roles": ["l_out"], "positions": [1],
            "top_k": 3, "byte_cap": 4096,
        }

    def trace(self, precision, out="trace", rules=None, check=True, arch="llama", prompt="abc",
              env=None):
        model = self.work / f"{arch}-{precision}.gguf"
        if not model.exists():
            write_model(model, precision, arch)
        spec = self.work / "spec.json"
        spec.write_text(json.dumps(self.spec))
        output = self.work / out
        args = ["trace", "--model", model, "--prompt", prompt, "--spec", spec, "--out", output]
        if rules is not None:
            interventions = self.work / "interventions.json"
            interventions.write_text(json.dumps(rules))
            args += ["--intervention", interventions]
        result = self.run_cli(*args, env=env)
        if not check:
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertFalse(output.exists())
            return result.stderr
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads((output / "manifest.json").read_text())

    def run_cli(self, *args, env=None):
        return subprocess.run(
            [self.binary, *args, "--threads", "2"],
            capture_output=True, text=True, env=env,
        )

    def tensor(self, entry, out="trace"):
        data = (self.work / out / entry["file"]).read_bytes()
        self.assertEqual(entry["sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(entry["bytes"], len(data))
        return struct.unpack(f"<{len(data) // 4}f", data)

    def test_selected_residual_rows_and_logits(self):
        # This cannot be a trace-store directory. Structured trace must ignore it.
        bad_home = self.work / "not-a-directory"
        bad_home.write_text("")
        manifest = self.trace("f32", env=dict(os.environ, LANTERNFISH_HOME=str(bad_home)))
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["tokens"][1]["position"], 1)
        nodes = manifest["tensors"]
        self.assertEqual([(n["layer"], n["role"], n["position"]) for n in nodes],
                         [(0, "l_out", 1), (1, "l_out", 1)])
        for node in nodes:
            self.assertEqual(node["shape"], [32])
            self.assertTrue(any(self.tensor(node)))
        self.assertEqual(len(manifest["model_logits"][0]["top"]), 3)
        self.assertEqual([row["layer"] for row in manifest["logit_lens"]], [0, 1])
        for lens, actual in zip(manifest["logit_lens"][-1]["top"],
                                manifest["model_logits"][0]["top"]):
            self.assertEqual(lens["token_id"], actual["token_id"])
            self.assertAlmostEqual(lens["logit"], actual["logit"], places=5)

    def test_legacy_run_still_writes_viz_json(self):
        model = self.work / "legacy.gguf"
        write_model(model, "f32")
        home = self.work / "store"
        result = self.run_cli("run", model, "--prompt", "abc", "--name", "legacy",
                              "-n", "0", "--raw", "--home", home)
        self.assertEqual(result.returncode, 0, result.stderr)
        path = Path(json.loads(result.stdout)["path"])
        self.assertEqual(path.name, "viz.json")
        self.assertTrue(path.is_file())

    def test_write_back_on_bf16_and_q8_qwen2(self):
        self.spec.update(layers=[1], roles=["attn_out", "ffn_inp", "ffn_out", "l_out", "kq_soft_max"],
                         positions=[0, 1, 2, 3], byte_cap=65536)
        vector = [0.125] * 32
        rules = [
            {"layer": 1, "role": "attn_out", "positions": [0], "op": "add", "vector": vector},
            {"layer": 1, "role": "ffn_inp", "positions": [1], "op": "scale", "factor": -2},
            {"layer": 1, "role": "ffn_out", "positions": [2], "op": "zero"},
            {"layer": 1, "role": "l_out", "positions": [3], "op": "replace", "vector": vector},
        ]
        for precision in ["bf16", "q8_0"]:
            with self.subTest(precision=precision):
                base = self.trace(precision, f"base-{precision}", arch="qwen2")
                edit = self.trace(precision, f"edit-{precision}", rules=rules, arch="qwen2")
                self.assertEqual(edit["missing"], [])
                self.assertEqual(len(edit["interventions"]), 4)

                def rows(manifest, out):
                    return {(n["role"], n["position"]): self.tensor(n, out)
                            for n in manifest["tensors"]}

                before, after = rows(base, f"base-{precision}"), rows(edit, f"edit-{precision}")
                for x, y in zip(before["attn_out", 0], after["attn_out", 0]):
                    self.assertAlmostEqual(y, x + 0.125, places=6)
                for x, y in zip(before["ffn_inp", 0], after["ffn_inp", 0]):
                    self.assertAlmostEqual(y, x + 0.125, places=6)
                self.assertEqual(after["ffn_inp", 1], tuple(-2 * x for x in before["ffn_inp", 1]))
                self.assertEqual(after["ffn_out", 2], (0.0,) * 32)
                self.assertEqual(after["l_out", 2], after["ffn_inp", 2])
                self.assertEqual(after["l_out", 3], tuple(vector))
                self.assertNotEqual(base["model_logits"], edit["model_logits"])
                attn = next(n for n in edit["tensors"] if n["role"] == "kq_soft_max")
                heads, keys = attn["shape"]
                self.assertEqual(heads, 4)
                self.assertGreaterEqual(keys, 4)
                for head in range(heads):
                    self.assertAlmostEqual(sum(after["kq_soft_max", 0][head*keys:(head+1)*keys]), 1)
                    self.assertEqual(after["kq_soft_max", 0][head*keys + 1], 0)

    def test_direction_interventions_on_selected_qwen2_rows(self):
        self.spec.update(layers=[1], positions=[0, 1, 2], byte_cap=8192)
        direction = [2.0, -1.0] + [0.0] * 30
        base = self.trace("bf16", "base", arch="qwen2")
        common = {"layer": 1, "role": "l_out", "vector": direction}
        edit = self.trace("bf16", "edit", arch="qwen2", rules=[
            dict(common, positions=[0], op="project_out"),
            dict(common, positions=[2], op="subtract", scale=0.25),
        ])
        before = [self.tensor(entry, "base") for entry in base["tensors"]]
        after = [self.tensor(entry, "edit") for entry in edit["tensors"]]
        self.assertGreater(abs(2 * before[0][0] - before[0][1]), 1e-5)
        self.assertAlmostEqual(2 * after[0][0] - after[0][1], 0, places=6)
        self.assertAlmostEqual(after[0][0] + 2 * after[0][1],
                               before[0][0] + 2 * before[0][1], places=6)
        self.assertEqual(after[0][2:], before[0][2:])
        self.assertEqual(after[1], before[1])
        for observed, original, d in zip(after[2], before[2], direction):
            self.assertAlmostEqual(observed, original - 0.25 * d, places=6)
        self.assertNotEqual(edit["model_logits"], base["model_logits"])
        self.assertIn("nonzero", self.trace("bf16", "invalid", arch="qwen2", check=False,
                                            rules=[dict(common, positions=[0], op="project_out",
                                                        vector=[0.0] * 32)]))

    def test_byte_cap_includes_manifest_and_tensors(self):
        self.spec["byte_cap"] = 256  # the two residuals fit, the manifest does not
        self.assertIn("byte_cap", self.trace("f32", check=False))

    def test_full_logits_and_scaled_projection(self):
        self.spec.update(layers=[], roles=[], positions=[1], top_k=3, byte_cap=65536)
        top = self.trace("bf16", "top", arch="qwen2")["model_logits"][0]["top"]
        self.spec.update(top_k=0, all_logits=True)
        baseline = self.trace("bf16", "full", arch="qwen2")["model_logits"][0]["top"]
        self.assertEqual([row["token_id"] for row in baseline], list(range(len(baseline))))
        self.assertEqual(top, sorted(baseline, key=lambda row: (-row["logit"], row["token_id"]))[:3])
        rule = {"layer": 1, "role": "l_out", "positions": [1],
                "op": "project_out", "vector": [1.] + [0.] * 31}
        plain = self.trace("bf16", "default", arch="qwen2", rules=[rule])
        unit = self.trace("bf16", "unit", arch="qwen2", rules=[dict(rule, scale=1)])
        scaled = self.trace("bf16", "scaled", arch="qwen2", rules=[dict(rule, scale=1.5)])
        self.assertEqual(plain["model_logits"], unit["model_logits"])
        self.assertNotEqual(plain["model_logits"], scaled["model_logits"])

    def test_lens_only_does_not_export_or_charge_residual_payloads(self):
        self.spec.update(roles=[], top_k=1, byte_cap=1000)
        manifest = self.trace("f32")
        self.assertEqual(manifest["tensors"], [])
        self.assertEqual(len(manifest["logit_lens"]), 2)

    def test_intervention_position_stays_absolute_across_batches(self):
        self.spec.update(roles=["l_out"], layers=[1], positions=[31, 32, 33], top_k=0,
                         byte_cap=8192)
        prompt = "a" * 40
        base = self.trace("f32", "base", prompt=prompt)
        edit = self.trace("f32", "edit", prompt=prompt, rules=[
            {"layer": 1, "role": "l_out", "positions": [32], "op": "zero"},
        ])
        before = [self.tensor(entry, "base") for entry in base["tensors"]]
        after = [self.tensor(entry, "edit") for entry in edit["tensors"]]
        self.assertTrue(any(before[1]))
        self.assertEqual(after, [before[0], (0.0,) * 32, before[2]])

    def test_logit_lens_includes_optional_output_bias(self):
        manifest = self.trace("f32", arch="qwen2")
        for lens, actual in zip(manifest["logit_lens"][-1]["top"],
                                manifest["model_logits"][0]["top"]):
            self.assertEqual(lens["token_id"], actual["token_id"])
            self.assertAlmostEqual(lens["logit"], actual["logit"], places=5)

    def test_grad_rejects_quantized_and_bf16_weights(self):
        for precision in ("bf16", "q8_0"):
            with self.subTest(precision=precision):
                model = self.work / f"{precision}.gguf"
                write_model(model, precision)
                out = self.work / f"grad-{precision}"
                result = self.run_cli(
                    "grad", "--model", model, "--prompt", "abc", "--layer", "0",
                    "--target-token", "6", "--out", out,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("requires F32", result.stderr)
                self.assertFalse(out.exists())

    def test_gradient_matches_finite_differences_at_selected_layers(self):
        target_id = 6
        self.spec.update(layers=[], roles=[], positions=[4], top_k=263, byte_cap=65536)
        for arch in ("llama", "qwen2"):
            model = self.work / f"{arch}-f32.gguf"
            for layer in (0, 1):
                with self.subTest(arch=arch, layer=layer):
                    write_model(model, "f32", arch)
                    out = f"grad-{arch}-{layer}"
                    result = self.run_cli(
                        "grad", "--model", model, "--prompt", "abc", "--layer", str(layer),
                        "--target-token", str(target_id), "--out", self.work / out,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    manifest = json.loads((self.work / out / "manifest.json").read_text())
                    self.assertEqual(manifest["schema_version"], 1)
                    self.assertEqual(manifest["kind"], "gradient")
                    self.assertEqual(manifest["tensor"]["shape"], [32, 64])
                    grad = self.tensor(manifest["tensor"], out)
                    index = max(range(len(grad)), key=lambda i: abs(grad[i]))
                    self.assertGreater(abs(grad[index]), 0.001)
                    step = 0.003
                    name = f"blk.{layer}.ffn_down.weight"
                    losses = []
                    for suffix, delta in (("base", 0), ("plus", step), ("minus", -step)):
                        write_model(model, "f32", arch, (name, index, delta))
                        captured = self.trace("f32", f"{out}-{suffix}", arch=arch)
                        logits = {r["token_id"]: r["logit"]
                                  for r in captured["model_logits"][0]["top"]}
                        losses.append(math.log(sum(math.exp(v) for v in logits.values()))
                                      - logits[target_id])
                    self.assertAlmostEqual(manifest["loss"], losses[0], delta=0.00005)
                    self.assertAlmostEqual(grad[index], (losses[1] - losses[2]) / (2 * step),
                                           delta=0.0003)


if __name__ == "__main__":
    unittest.main()
