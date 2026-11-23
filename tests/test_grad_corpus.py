"""Compare the token-weighted corpus objective with independent prefix calls."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

from tiny_gguf import write_model


class CorpusGradient(unittest.TestCase):
    def test_prefix_average_and_finite_difference(self):
        binary = os.environ["LANTERNFISH_BIN"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, spec, corpus = (root / k for k in ("m.gguf", "spec", "corpus"))
            write_model(model, "f32", "qwen2")
            corpus.write_text("a\nabcab\n")
            digest = hashlib.sha256(corpus.read_bytes()).hexdigest()
            names = ["blk.0.attn_q.bias", "blk.1.ffn_down.weight"]

            def run(cmd, dest, body, *args):
                out = root / dest
                spec.write_text(json.dumps(body))
                cli = binary
                env = os.environ.copy()
                if cmd == "grad-select" and "X3_PIN" in env:
                    cli = Path(env["X3_PIN"]) / "lanternfish"
                    env["LD_LIBRARY_PATH"] = str(cli.parent / "lib/cpu")
                if cmd == "grad-corpus":
                    args = (*args, "--backend", env.get("CORPUS_TEST_BACKEND", "cpu"))
                subprocess.run([cli, cmd, "--model", model, "--threads", "2",
                                "--spec", spec, "--out", out, *args], check=True, env=env)
                return json.loads((out / "gradient.json").read_bytes())

            def batch(dest):
                return run("grad-corpus", dest, dict(corpus_sha256=digest, weights=names),
                           "--corpus", corpus)

            def floats(p):
                return [x for x, in struct.iter_unpack("<f", p.read_bytes())]

            meta = batch("joint")
            ids = [json.loads(subprocess.check_output([binary, "tokenize", "--model", model,
                   "--text", line]))["tokens"] for line in corpus.read_text().splitlines()]
            count = sum(len(seq)-1 for seq in ids)
            self.assertEqual(meta["tokens"], ids)
            self.assertEqual(meta["corpus_sha256"], digest)
            self.assertEqual(meta["observation_count"], count)
            self.assertEqual([g["name"] for g in meta["gradients"]], names)
            for row in meta["gradients"]:
                path = root / "joint" / row["file"]
                self.assertEqual(row["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                got = floats(path)
                ref = [0.] * len(got)
                loss = 0.
                for j, seq in enumerate(ids):
                    for k in range(1, len(seq)):
                        dest = f'{row["name"]}-{j}-{k}'
                        single = run("grad-select", dest, dict(tokens=seq[:k], target_id=seq[k],
                                     site=dict(kind="weight", name=row["name"])))
                        loss += single["loss"] / count
                        for i, v in enumerate(floats(root / dest / "gradient.f32")):
                            ref[i] += v / count
                self.assertAlmostEqual(loss, meta["loss"], delta=2e-5)
                self.assertLess(max(abs(a-b) for a,b in zip(got, ref)), 2e-5)
            i = max(range(len(got)), key=lambda k: abs(got[k]))
            loss = []
            for sign in (-1, 1):
                write_model(model, "f32", "qwen2", (names[-1], i, sign*.003))
                loss.append(batch(str(sign))["loss"])
            self.assertAlmostEqual(got[i], (loss[1]-loss[0])/.006, delta=3e-4)
