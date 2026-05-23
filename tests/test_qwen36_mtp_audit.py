import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parent / "metal" / "audit_qwen36_mtp.py"
SPEC = importlib.util.spec_from_file_location("audit_qwen36_mtp", SCRIPT)
audit_qwen36_mtp = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = audit_qwen36_mtp
SPEC.loader.exec_module(audit_qwen36_mtp)


class Qwen36MtpAuditTests(unittest.TestCase):
    def test_audit_required_reports_complete_partial_and_absent(self):
        required = ["a", "b", "c"]
        complete = audit_qwen36_mtp.audit_required({"a", "b", "c"}, required)
        partial = audit_qwen36_mtp.audit_required({"a"}, required)
        absent = audit_qwen36_mtp.audit_required(set(), required)

        self.assertEqual(complete["status"], "complete")
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(partial["missing"], ["b", "c"])
        self.assertEqual(absent["status"], "absent")

    def test_source_audit_accepts_split_expert_projection_names(self):
        names = set(audit_qwen36_mtp.SOURCE_EXACT_TENSORS)
        for expert in range(2):
            names.add(f"mtp.layers.0.mlp.experts.{expert}.gate_proj.weight")
            names.add(f"mtp.layers.0.mlp.experts.{expert}.up_proj.weight")
            names.add(f"mtp.layers.0.mlp.experts.{expert}.down_proj.weight")

        source = audit_qwen36_mtp.audit_source(
            names,
            {"status": "loaded", "tensor_count": len(names), "mtp_tensor_count": len(names)},
            num_experts=2,
        )

        self.assertEqual(source["status"], "complete")
        for row in source["expert_projection_tensors"]:
            self.assertEqual(row["status"], "complete")
            self.assertEqual(row["missing_experts"], [])

    def test_load_bake_manifest_and_loader_delta_ready(self):
        with tempfile.TemporaryDirectory() as td:
            bake_dir = Path(td)
            manifest = {
                "format_version": 2,
                "converter_version": 1,
                "quant_profile": "int4-gptq",
                "tensors": [
                    {
                        "name": name,
                        "shape": [1],
                        "dtype": "BF16",
                        "layout": "row_major",
                        "byte_len": 2,
                    }
                    for name in audit_qwen36_mtp.REQUIRED_BAKE_TENSORS
                ],
            }
            (bake_dir / "manifest.json").write_text(json.dumps(manifest))

            names, meta = audit_qwen36_mtp.load_bake_manifest(bake_dir)
            bake = audit_qwen36_mtp.audit_bake(names, meta)
            delta = audit_qwen36_mtp.loader_delta({"status": "complete"}, bake)

            self.assertEqual(bake["status"], "complete")
            self.assertTrue(bake["runtime_probe_present"])
            self.assertEqual(delta["status"], "ready")
            self.assertIn("mtp.fc.weight", bake["mtp_tensor_metadata"])

    def test_loader_delta_marks_partial_bake_as_fail_closed(self):
        names = {"mtp.fc.weight"}
        bake = audit_qwen36_mtp.audit_bake(names, {"status": "loaded"})
        delta = audit_qwen36_mtp.loader_delta({"status": "complete"}, bake)

        self.assertEqual(bake["status"], "partial")
        self.assertEqual(delta["status"], "partial_bake")
        self.assertIn("runtime loader will fail closed", " ".join(delta["notes"]))

    def test_render_markdown_includes_statuses_and_missing_tensors(self):
        report = {
            "model": audit_qwen36_mtp.MODEL,
            "model_dir": "/models/qwen",
            "bake_dir": "/models/qwen/.supersonic/v2-int4-gptq",
            "source": {
                "status": "partial",
                "tensor_count": 10,
                "mtp_tensor_count": 4,
                "exact_tensors": {"present_count": 1, "missing": ["mtp.norm.weight"]},
                "expert_projection_tensors": [
                    {
                        "projection": "gate_proj",
                        "status": "partial",
                        "present_count": 1,
                        "required_count": 2,
                        "missing_experts": [1],
                    }
                ],
            },
            "bake": {
                "status": "partial",
                "tensor_count": 2,
                "mtp_tensor_count": 1,
                "required_tensors": {
                    "present_count": 1,
                    "missing": ["mtp.layers.0.mlp.experts.down_proj"],
                },
            },
            "loader_delta": {
                "status": "partial_bake",
                "notes": ["Missing bake tensors: mtp.layers.0.mlp.experts.down_proj"],
            },
        }

        md = audit_qwen36_mtp.render_markdown(report)

        self.assertIn("loader status: `partial_bake`", md)
        self.assertIn("`mtp.layers.0.mlp.experts.down_proj`", md)
        self.assertIn("| `gate_proj` | `partial` | 1/2 | 1 |", md)


if __name__ == "__main__":
    unittest.main()
