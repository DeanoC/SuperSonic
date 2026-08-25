import importlib
import unittest


def load_quality_module():
    try:
        return importlib.import_module("tools.benchmark.quality")
    except ModuleNotFoundError as exc:
        raise AssertionError("tools.benchmark.quality is absent") from exc


def load_benchmark_api():
    try:
        return importlib.import_module("tools.benchmark")
    except ModuleNotFoundError as exc:
        raise AssertionError("tools.benchmark package is absent") from exc


class BenchmarkQualityTests(unittest.TestCase):
    maxDiff = None

    def case(
        self,
        expected,
        *,
        scorer="exact_text",
        case_id="case-1",
        category="instruction-following",
    ):
        benchmark = load_benchmark_api()
        return benchmark.QualityCase(
            id=case_id,
            category=category,
            prompt="prompt",
            max_new_tokens=8,
            scorer=scorer,
            expected=expected,
            decoding_policy="greedy",
        )

    def output(
        self,
        generated_text,
        *,
        token_ids=(10,),
        engine_name="supersonic",
    ):
        benchmark = load_benchmark_api()
        generated_count = len(token_ids) if token_ids is not None else 1
        return benchmark.ParsedOutput(
            engine_name=engine_name,
            engine_version="test",
            generated_text=generated_text,
            token_ids=token_ids,
            prompt_tokens=4,
            generated_tokens=generated_count,
            decode_ms=12.0,
            ms_per_tok=12.0 / generated_count,
            tokens_per_second=generated_count / 0.012,
        )

    def matching_output_for_case(self, case):
        benchmark = load_benchmark_api()
        if case.scorer == "exact_text":
            return self.output(case.expected)
        if case.scorer == "exact_tokens":
            return self.output("unused", token_ids=tuple(case.expected))
        if case.scorer == "structured_json":
            return self.output(benchmark.canonical_json(case.expected))
        raise AssertionError(f"unsupported test scorer: {case.scorer}")

    def test_exact_text_is_not_fuzzy(self):
        quality = load_quality_module()

        result = quality.score_case(self.case("42"), self.output("42 "))

        self.assertFalse(result.passed)
        self.assertIn("exact", result.failure.lower())

    def test_exact_tokens_fail_clearly_when_token_ids_are_unavailable(self):
        quality = load_quality_module()

        result = quality.score_case(self.case([1, 2, 3], scorer="exact_tokens"), self.output("1 2 3", token_ids=None))

        self.assertFalse(result.passed)
        self.assertIn("token", result.failure.lower())
        self.assertIn("unavailable", result.failure.lower())

    def test_structured_json_compares_values_without_format_fuzziness(self):
        quality = load_quality_module()

        result = quality.score_case(
            self.case({"answer": 42, "mode": "ordinary"}, scorer="structured_json"),
            self.output('{ "mode": "ordinary", "answer": 42 }'),
        )

        self.assertTrue(result.passed)

    def test_structured_json_uses_parsed_value_equality_for_numeric_json(self):
        quality = load_quality_module()

        result = quality.score_case(
            self.case({"answer": 42}, scorer="structured_json"),
            self.output('{"answer": 42.0}'),
        )

        self.assertTrue(result.passed)

    def test_structured_json_rejects_duplicate_keys(self):
        quality = load_quality_module()

        result = quality.score_case(
            self.case({"answer": 42}, scorer="structured_json"),
            self.output('{"answer": 1, "answer": 42}'),
        )

        self.assertFalse(result.passed)
        self.assertIn("duplicate", result.failure.lower())

    def test_structured_json_rejects_non_finite_numbers(self):
        quality = load_quality_module()

        result = quality.score_case(
            self.case({"answer": 42}, scorer="structured_json"),
            self.output('{"answer": 1e309}'),
        )

        self.assertFalse(result.passed)
        self.assertIn("non-finite", result.failure.lower())

    def test_mtp_requires_identical_tokens(self):
        quality = load_quality_module()

        result = quality.score_mtp_pair(
            self.output("same", token_ids=(1, 2)),
            self.output("same", token_ids=(1, 3)),
            case_id="mtp-case-1",
            category="ordinary-vs-mtp-token-equality",
        )

        self.assertFalse(result.passed)
        self.assertIn("token", result.failure.lower())

    def test_mtp_pair_evidence_uses_the_ordinary_tokens_it_compares(self):
        quality = load_quality_module()
        case = self.case(
            [1, 2, 3],
            scorer="exact_tokens",
            case_id="mtp-evidence",
            category="ordinary-vs-mtp-token-equality",
        )

        result = quality.score_mtp_pair(
            self.output("ordinary", token_ids=(40, 4021, 3300)),
            self.output("mtp", token_ids=(40, 4021, 3300)),
            case=case,
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.expected_value, "[40,4021,3300]")
        self.assertEqual(result.actual_value, "[40,4021,3300]")
        self.assertEqual(result.expected_hash, result.actual_hash)

    def test_mtp_fails_clearly_when_tokens_are_unavailable(self):
        quality = load_quality_module()

        result = quality.score_mtp_pair(
            self.output("ordinary", token_ids=None),
            self.output("mtp", token_ids=(1, 2)),
            case_id="mtp-case-2",
            category="ordinary-vs-mtp-token-equality",
        )

        self.assertFalse(result.passed)
        self.assertIn("token", result.failure.lower())
        self.assertIn("unavailable", result.failure.lower())

    def test_mtp_pair_requires_manifest_case_or_explicit_identity(self):
        quality = load_quality_module()

        with self.assertRaisesRegex(ValueError, "requires a manifest QualityCase or explicit case_id/category"):
            quality.score_mtp_pair(self.output("ordinary", token_ids=(1, 2)), self.output("mtp", token_ids=(1, 2)))

    def test_mtp_pair_requires_unique_manifest_case_identity_for_summary(self):
        manifest = importlib.import_module("tools.benchmark.manifest")
        quality = load_quality_module()

        mtp_cases = tuple(
            case for case in manifest.load_quality("v1") if case.category == "ordinary-vs-mtp-token-equality"
        )
        self.assertEqual(len(mtp_cases), 2)

        results = tuple(
            quality.score_mtp_pair(
                self.output("ordinary", token_ids=tuple(case.expected)),
                self.output("mtp", token_ids=tuple(case.expected)),
                case=case,
            )
            for case in mtp_cases
        )
        summary = quality.summarize_quality(results, required_cases=mtp_cases)

        self.assertEqual(summary["failed"], 0)
        self.assertEqual(summary["passed"], 2)
        self.assertEqual(summary["missing_case_ids"], [])
        self.assertEqual(
            [entry["id"] for entry in summary["cases"]],
            [case.id for case in mtp_cases],
        )

    def test_duplicate_quality_result_ids_fail_closed(self):
        quality = load_quality_module()

        with self.assertRaisesRegex(ValueError, "duplicate quality result id"):
            quality.summarize_quality(
                (
                    quality.score_mtp_pair(
                        self.output("ordinary", token_ids=(1, 2, 3)),
                        self.output("mtp", token_ids=(1, 2, 3)),
                        case_id="duplicate",
                        category="ordinary-vs-mtp-token-equality",
                    ),
                    quality.score_mtp_pair(
                        self.output("ordinary", token_ids=(4, 5, 6)),
                        self.output("mtp", token_ids=(4, 5, 6)),
                        case_id="duplicate",
                        category="ordinary-vs-mtp-token-equality",
                    ),
                )
            )

    def test_summarize_quality_reports_every_category_and_preserves_evidence(self):
        manifest = importlib.import_module("tools.benchmark.manifest")
        quality = load_quality_module()

        cases = manifest.load_quality("v1")
        results = tuple(quality.score_case(case, self.matching_output_for_case(case)) for case in cases)
        summary = quality.summarize_quality(results, required_cases=cases)

        self.assertEqual(summary["passed"], len(cases))
        self.assertEqual(summary["failed"], 0)
        self.assertEqual(summary["total"], len(cases))
        self.assertEqual(set(summary["categories"]), manifest.APPROVED_CATEGORIES)
        self.assertEqual(summary["categories"]["repeated-run-determinism"], {"passed": 2, "failed": 0, "total": 2})
        self.assertEqual(summary["categories"]["ordinary-vs-mtp-token-equality"], {"passed": 2, "failed": 0, "total": 2})
        self.assertEqual(summary["missing_case_ids"], [])
        self.assertEqual(len(summary["cases"]), len(cases))
        first_case = summary["cases"][0]
        self.assertEqual(set(first_case), {
            "id",
            "category",
            "scorer",
            "passed",
            "failure",
            "expected_hash",
            "actual_hash",
            "expected_value",
            "actual_value",
        })
        self.assertEqual(len(first_case["expected_hash"]), 64)
        self.assertEqual(len(first_case["actual_hash"]), 64)

    def test_summarize_quality_counts_missing_required_cases_as_failures(self):
        manifest = importlib.import_module("tools.benchmark.manifest")
        quality = load_quality_module()

        cases = manifest.load_quality("v1")
        available_cases = cases[:-1]
        results = tuple(quality.score_case(case, self.matching_output_for_case(case)) for case in available_cases)

        summary = quality.summarize_quality(results, required_cases=cases)

        self.assertEqual(summary["passed"], len(available_cases))
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["total"], len(cases))
        self.assertEqual(summary["missing_case_ids"], [cases[-1].id])
        self.assertFalse(summary["cases"][-1]["passed"])
        self.assertIn("missing", summary["cases"][-1]["failure"].lower())

    def test_any_required_case_failure_fails_the_aggregate(self):
        manifest = importlib.import_module("tools.benchmark.manifest")
        quality = load_quality_module()

        cases = manifest.load_quality("v1")
        failed_case_id = "repeated-run-determinism-2"
        results = []
        for case in cases:
            output = self.matching_output_for_case(case)
            if case.id == failed_case_id:
                output = self.output("5678 ")
            results.append(quality.score_case(case, output))

        summary = quality.summarize_quality(tuple(results), required_cases=cases)

        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["categories"]["repeated-run-determinism"], {"passed": 1, "failed": 1, "total": 2})
        failed_entries = [case for case in summary["cases"] if not case["passed"]]
        self.assertEqual([entry["id"] for entry in failed_entries], [failed_case_id])


if __name__ == "__main__":
    unittest.main()
