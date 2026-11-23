import math
import unittest

from w14b_acceptance import compare, cross_entropy


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
