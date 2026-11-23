"""Run the tiny whole-graph CLI regression inside the GPU queue."""
from argparse import Namespace
import json
import os
from pathlib import Path
import re
import unittest

from w14b_acceptance import audit_log, check_backend, load_json, run_cli, write_model


@unittest.skipUnless(os.environ.get("W14B_PLACEMENT_EVIDENCE"), "requires a GPU queue turn")
class PlacementTests(unittest.TestCase):
    def test_gpu_cli_has_no_cpu_compute_in_dispatched_graphs(self):
        backend = os.environ.get("W14B_PLACEMENT_BACKEND", "cuda")
        root = Path(os.environ["W14B_PLACEMENT_EVIDENCE"])
        root.mkdir()
        args = Namespace(binary=Path(os.environ["LANTERNFISH_BIN"]), evidence=root,
                         threads=2, cpu_only=False)
        model = root / "tiny.gguf"
        write_model(model, "f32", "qwen2")
        request = root / "rome-request.json"
        request.write_text(json.dumps({
            "method": "rome", "layers": [0], "steps": 2,
            "facts": [{"subject": "abc", "prompt": "{}d", "target": " ef"}],
        }))
        commands = {
            "grad": ("--prompt", "abc", "--layer", "0", "--target-token", "6",
                     "--out", root / "gradient"),
            "locate-edit": ("--request", request, "--out", root / "rome.json",
                            "--cache", root / "cache"),
        }
        for command, options in commands.items():
            with self.subTest(command=command):
                run_cli(args, command, command, "--model", model, "--backend", backend,
                        *options, audit=True, scheduler_debug=True)
                # The final scheduler plan follows the last reserve. A real grad
                # or edit must run it. Ignore unrelated reserve plans and c=0 nodes.
                log = (root / f"{command}.log").read_text()
                plan = log.rsplit("sched_reserve: reserve took", 1)[-1]
                self.assertIn("## SPLIT", plan)
                fallback = re.findall(
                    r"^node #.*?\[ *CPU +\] use=\d+,c=1:",
                    plan, re.MULTILINE,
                )
                self.assertFalse(fallback, f"executed CPU compute: {fallback}")
                if command == "grad":
                    manifest = root / "gradient" / "manifest.json"
                    required = {"backward"}
                else:
                    manifest = root / "rome.json"
                    required = {"forward", "backward"}
                device = check_backend(load_json(manifest), backend)["device"]
                audit_log(root, command, device, required)
