import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from w14b_acceptance import compare, cross_entropy, fp64_oracle, gradient_parity, placement


class GradientComparisonTests(unittest.TestCase):
    def test_absolute_floor_and_relative_scale_both_apply(self):
        stats = compare([0.0, -10.0], [0.005, -10.25], atol=0.01, rtol=0.05)
        self.assertTrue(stats["passed"])
        self.assertEqual(stats["max_absolute_error"], 0.25)
        self.assertEqual(stats["relative_linf_error"], 0.025)
        self.assertEqual(stats["max_tolerance_ratio"], 0.5)

    def test_large_reference_does_not_hide_error_at_zero(self):
        stats = compare([0.0, 100.0], [0.015, 100.0], atol=0.01, rtol=0.05)
        self.assertFalse(stats["passed"])
        self.assertEqual(stats["max_tolerance_ratio"], 1.5)

    def test_bad_inputs_never_pass(self):
        for ref, actual in (([1], []), ([], []), ([1], [math.nan]), ([math.inf], [1])):
            with self.subTest(ref=ref, actual=actual), self.assertRaises(ValueError):
                compare(ref, actual, atol=0.01, rtol=0.05)

    def test_trace_loss_uses_all_tokens_and_is_stable_for_large_logits(self):
        top = [{"token_id": token, "logit": 1000.0} for token in (2, 1, 0)]
        self.assertAlmostEqual(cross_entropy(top, target=1, vocab_size=3), math.log(3))
        with self.assertRaises(ValueError):
            cross_entropy(top[:-1], target=1, vocab_size=3)

    def test_tensor_acceptance_does_not_relabel_pointwise_failure(self):
        stats = gradient_parity([0.0, 1.0], [0.0005, 1.0], 4.0, 4.0)
        self.assertFalse(stats["pointwise_passed"])
        self.assertTrue(stats["relative_linf_passed"])
        self.assertTrue(stats["loss_passed"])
        self.assertFalse(gradient_parity([1.0], [1.01], 4.0, 4.1)["relative_linf_passed"])
        self.assertFalse(gradient_parity([1.0], [1.0], 4.0, 4.1)["loss_passed"])
        self.assertFalse(gradient_parity([0.0], [0.0], 4.0, 4.0)["relative_linf_passed"])


class PlacementAuditTests(unittest.TestCase):
    def test_audit_requires_actual_dispatch_on_the_selected_device(self):
        reserve = "## SPLIT #0: CPU # 0 inputs\n"
        backward = "lanternfish_placement: graph=backward device=CUDA0 compute_ops=8 cpu_compute_ops=0\n"
        forward = "lanternfish_placement: graph=forward device=CUDA0 compute_ops=3 cpu_compute_ops=0\n"
        required = {"forward", "backward"}
        self.assertEqual(
            placement(reserve + forward + backward + forward, "CUDA0", required),
            {"forward": {"graphs": 2, "compute_ops": 6, "cpu_compute_ops": 0},
             "backward": {"graphs": 1, "compute_ops": 8, "cpu_compute_ops": 0}},
        )
        for log in (reserve, reserve + backward, forward + backward.replace("CUDA0", "CUDA1"),
                    forward + backward.replace("cpu_compute_ops=0", "cpu_compute_ops=1"),
                    forward + backward.replace("compute_ops=8", "compute_ops=0"),
                    forward + backward + "lanternfish_placement: graph=bad"):
            with self.subTest(log=log), self.assertRaises(ValueError):
                placement(log, "CUDA0", required)


class OracleTests(unittest.TestCase):
    identity = {
        "binary_sha256": "a" * 64,
        "model_sha256": "b" * 64,
        "gradient_sha256": {"cpu": "c" * 64, "cuda": "d" * 64},
    }

    def write_oracle(self, path, cases):
        path.write_text(json.dumps({"schema_version": 1, "cases": cases, **self.identity}))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_pinned_operand_set_must_pass_both_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oracle.json"
            for cpu, cuda, expected in ((8e-6, 1e-6, True), (2e-5, 2e-5, False),
                                         (1e-7, 2e-7, False)):
                with self.subTest(cpu=cpu, cuda=cuda):
                    row = {"operand_set": "qwen-layer-23.out_proj",
                           "cpu_max_abs_error": cpu, "cuda_max_abs_error": cuda}
                    pin = self.write_oracle(path, [row])
                    result = fp64_oracle(path, pin, self.identity)
                    self.assertEqual(result["passed"], expected)
                    self.assertEqual(result["cases"], [dict(row, passed=expected)])

    def test_gate_fails_on_bad_pin_missing_or_invalid_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oracle.json"
            with self.assertRaises(OSError):
                fp64_oracle(path, "0" * 64, self.identity)
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                fp64_oracle(path, "0" * 64, self.identity)
            row = {"operand_set": "qwen-layer-23.out_proj",
                   "cpu_max_abs_error": 8e-6, "cuda_max_abs_error": 1e-6}
            for cases in ([], [dict(row, operand_set="other")], [dict(row, cuda_max_abs_error=-1.0)]):
                pin = self.write_oracle(path, cases)
                with self.assertRaises(ValueError):
                    fp64_oracle(path, pin, self.identity)
            pin = self.write_oracle(path, [row])
            with self.assertRaisesRegex(ValueError, "gradient_sha256"):
                fp64_oracle(path, pin, dict(self.identity, gradient_sha256={}))
