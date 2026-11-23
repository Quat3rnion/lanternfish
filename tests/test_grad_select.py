"""Exercise the selected-gradient CLI, including GGUF element order."""
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

from tiny_gguf import write_model


class SelectedGradient(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.model = self.root / "tiny.gguf"
        write_model(self.model, "f32", "qwen2")
        self.index = 0

    def run_site(self, site, **extra):
        self.index += 1
        spec = self.root / "spec.json"
        spec.write_text(json.dumps(dict(tokens=[1, 4, 5, 4], target_id=6, site=site) | extra))
        out = self.root / str(self.index)
        subprocess.run([os.environ["LANTERNFISH_BIN"], "grad-select", "--threads", "2",
                        "--model", self.model, "--spec", spec, "--out", out], check=True)
        meta = json.loads((out / "gradient.json").read_text())
        data = list(x for x, in struct.iter_unpack("<f", (out / "gradient.f32").read_bytes()))
        return meta, data

    def test_weight_difference(self):
        name = "blk.0.attn_q.bias"
        site = dict(kind="weight", name=name)
        meta, grad = self.run_site(site)
        self.assertEqual(meta["shape"], [32])
        self.assertEqual(meta["backend"], "cpu")
        i = max(range(len(grad)), key=lambda j: abs(grad[j]))
        losses = []
        for sign in (-1, 1):
            write_model(self.model, "f32", "qwen2", (name, i, sign * 0.003))
            losses.append(self.run_site(site)[0]["loss"])
        self.assertGreater(abs(grad[i]), 3e-4)
        self.assertAlmostEqual(grad[i], (losses[1]-losses[0])/0.006, delta=3e-4)

    def test_activation_derivatives_match_trace_interventions(self):
        binary = os.environ["LANTERNFISH_BIN"]
        prompt = "abc"
        tokens = json.loads(subprocess.check_output(
            [binary, "tokenize", "--model", self.model, "--text", prompt]))["tokens"]
        for role in ("attn_out", "ffn_out", "l_out"):
            # A non-final token at layer zero must propagate through later attention.
            pos = len(tokens) - 2
            site = dict(kind="activation", role=role, position=pos, layer=0)
            _, grad = self.run_site(site, tokens=tokens)
            i = max(range(len(grad)), key=lambda j: abs(grad[j]))
            self.assertGreater(abs(grad[i]), 3e-4)
            loss = []
            for sign in (-1, 1):
                spec, rule = self.root / "trace.json", self.root / "rule.json"
                spec.write_text(json.dumps(dict(layers=[0], roles=[role],
                    positions=[pos, len(tokens)-1], top_k=263, byte_cap=200000)))
                vec = [0.] * 32
                vec[i] = sign * 0.003
                rule.write_text(json.dumps([dict(layer=0, role=role, positions=[pos],
                                                  op="add", vector=vec)]))
                out = self.root / f"trace-{role}-{sign}"
                subprocess.run([binary, "trace", "--model", self.model, "--prompt", prompt,
                    "--spec", spec, "--intervention", rule, "--out", out, "--threads", "2"], check=True)
                doc = json.loads((out / "manifest.json").read_text())
                logits = {x["token_id"]: x["logit"] for x in doc["model_logits"][-1]["top"]}
                loss.append(math.log(sum(map(math.exp, logits.values()))) - logits[6])
            self.assertAlmostEqual(grad[i], (loss[1]-loss[0])/0.006, delta=3e-4)

    def test_tied_embedding_includes_lookup_and_head(self):
        write_model(self.model, "f32", tied=True)
        site = dict(kind="weight", name="token_embd.weight")
        meta, g = self.run_site(site)
        self.assertEqual(meta["shape"], [32, 263])
        # Restrict to a repeated input row: an output-only derivative misses
        # the lookup contribution; an embedding-only derivative misses head use.
        i = max(range(4*32, 5*32), key=lambda j: abs(g[j]))
        losses = []
        for sign in (-1, 1):
            write_model(self.model, "f32", perturb=(site["name"], i, sign * .001), tied=True)
            losses.append(self.run_site(site)[0]["loss"])
        self.assertAlmostEqual(g[i], (losses[1]-losses[0])/.002, delta=3e-4)

    def test_all_dense_qwen_weight_roles_and_sequence_edges(self):
        write_model(self.model, "bf16", "qwen2")
        for name in ("token_embd.weight", "output.weight", "output.bias", "output_norm.weight",
                     *(f"blk.0.{r}.weight" for r in ("attn_norm", "attn_q", "attn_k", "attn_v",
                        "attn_output", "ffn_norm", "ffn_gate", "ffn_up", "ffn_down")),
                     *(f"blk.0.attn_{r}.bias" for r in "qkv")):
            with self.subTest(name=name):
                meta, g = self.run_site(dict(kind="weight", name=name), tokens=[1])
                self.assertEqual(len(g), math.prod(meta["shape"]))
                self.assertTrue(all(map(math.isfinite, g)))
        # At the final block, an earlier FFN write cannot affect the last row.
        _, g = self.run_site(dict(kind="activation", layer=1, role="ffn_out", position=0),
                             tokens=[4, 5] * 64)
        self.assertEqual(g, [0.] * 32)
