from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "benchmark_fixtures" / "valid-result-v1.json"


def load_render_module():
    try:
        from tools.benchmark import render
    except ModuleNotFoundError as exc:  # pragma: no cover - RED phase guard
        raise AssertionError("tools.benchmark.render is absent") from exc
    return render


class BenchmarkRenderTests(unittest.TestCase):
    maxDiff = None

    def setUp(self) -> None:
        self.render = load_render_module()
        self.record = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def _write_results(self, root: Path, *, records: list[dict] | None = None) -> Path:
        results = root / "results"
        (results / "run-a" / "records").mkdir(parents=True)
        values = records if records is not None else [self.record]
        for index, record in enumerate(values):
            path = results / "run-a" / "records" / f"record-{index}.json"
            path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        return results

    @staticmethod
    def _snapshot(root: Path) -> dict[str, bytes]:
        return {
            str(path.relative_to(root)): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    def test_two_renders_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = self._write_results(root)
            first = root / "site-first"
            second = root / "site-second"

            self.render.render_site(results, first)
            self.render.render_site(results, second)

            self.assertEqual(self._snapshot(first), self._snapshot(second))

    def test_noncomparable_peer_has_reasons_and_no_speedup(self):
        peer = copy.deepcopy(self.record)
        peer["run"]["run_id"] = "peer-run"
        peer["engine"]["name"] = "peer"
        peer["environment"]["clock_policy"] = "uncontrolled-clocks"
        peer["hardware"]["clock_policy"] = "uncontrolled-clocks"
        peer["environment"]["headline_eligible"] = False
        peer["environment"]["verification_errors"] = [
            "headline eligibility requires locked clock_policy"
        ]

        page = self.render.render_comparison(self.record, peer)

        self.assertIn("uncontrolled-clocks", page)
        self.assertIn("clock_policy", page)
        self.assertNotIn("speedup", page.lower())

    def test_untrusted_text_is_escaped(self):
        record = copy.deepcopy(self.record)
        record["engine"]["version"] = "<script>alert(1)</script>"
        record["run"]["command"] = ["supersonic", "<script>alert(1)</script>"]

        page = self.render.render_run(record)

        self.assertNotIn("<script>", page)
        self.assertIn("&lt;script&gt;", page)

    def test_run_page_keeps_samples_statistics_evidence_and_failures(self):
        record = copy.deepcopy(self.record)
        record["quality"]["cases"][0]["passed"] = False
        record["quality"]["cases"][0]["failure"] = "expected <answer>"
        record["quality"]["failed"] = 1
        record["quality"]["passed"] = 7
        record["quality"]["categories"]["instruction-following"] = {
            "passed": 0,
            "failed": 1,
            "total": 1,
        }

        page = self.render.render_run(record)

        for value in ("30.0", "28.0", "32.0", "2.0", "3", "2400", "1249"):
            self.assertIn(value, page)
        for value in (
            "locked",
            "warm-resident",
            "quality",
            "expected &lt;answer&gt;",
            record["run"]["commit"],
            record["artifact"]["sha256"],
            record["workload"]["prompt_sha256"],
            "gfx1201",
            "reproduce",
            "supersonic",
            "v1",
        ):
            self.assertIn(value, page)
        self.assertNotIn("generation timestamp", page.lower())

    def test_site_has_expected_pages_and_filters_nonpublishable_records(self):
        incomplete = copy.deepcopy(self.record)
        incomplete["run"]["run_id"] = "incomplete-run"
        incomplete["status"]["state"] = "incomplete"
        incomplete["errors"] = [{"code": "budget_exhausted", "message": "budget"}]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = self._write_results(root, records=[self.record, incomplete])
            output = root / "site"

            paths = self.render.render_site(results, output)
            relative = {path.relative_to(output).as_posix() for path in paths}

            self.assertIn("index.html", relative)
            self.assertIn("methodology.html", relative)
            self.assertIn("assets/benchmarks.css", relative)
            self.assertTrue(any(path.startswith("runs/") for path in relative))
            self.assertTrue(any(path.startswith("trends/") for path in relative))
            self.assertTrue(any(path.startswith("comparisons/") for path in relative))
            site_text = "\n".join(path.read_text(encoding="utf-8") for path in output.rglob("*.html"))
            self.assertIn(self.record["run"]["run_id"], site_text)
            self.assertNotIn("incomplete-run", site_text)
            self.assertNotIn("<script", site_text.lower())

    def test_output_cleanup_requires_site_marker_and_does_not_delete_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = self._write_results(root)
            output = root / "site"
            output.mkdir()
            (output / "user-file.txt").write_text("keep", encoding="utf-8")

            with self.assertRaises(ValueError):
                self.render.render_site(results, output)
            self.assertTrue((output / "user-file.txt").exists())
            self.assertTrue((results / "run-a" / "records" / "record-0.json").exists())


if __name__ == "__main__":
    unittest.main()
