from __future__ import annotations

from dataclasses import dataclass
import hashlib

from .adapters import ParsedOutput
from .model import QualityCase, canonical_json, parse_strict_json


MTP_CATEGORY = "ordinary-vs-mtp-token-equality"
MAX_VALUE_PREVIEW = 160
_USE_CASE_EXPECTED = object()


@dataclass(frozen=True, slots=True)
class QualityResult:
    id: str
    category: str
    scorer: str
    passed: bool
    failure: str | None
    expected_hash: str
    actual_hash: str
    expected_value: str
    actual_value: str


def score_case(case: QualityCase, output: ParsedOutput) -> QualityResult:
    if case.scorer == "exact_text":
        actual = output.generated_text
        passed = actual == case.expected
        failure = None if passed else "generated text did not exactly match expected text"
        return _result(case, actual=actual, passed=passed, failure=failure)

    if case.scorer == "exact_tokens":
        if output.token_ids is None:
            return _result(
                case,
                actual=None,
                passed=False,
                failure="token ids unavailable for exact_tokens scorer",
            )
        actual = list(output.token_ids)
        passed = actual == case.expected
        failure = None if passed else "generated token ids did not exactly match expected token ids"
        return _result(case, actual=actual, passed=passed, failure=failure)

    if case.scorer == "structured_json":
        try:
            actual = parse_strict_json(output.generated_text, context="structured_json output")
        except ValueError as exc:
            return _result(case, actual=output.generated_text, passed=False, failure=str(exc))
        passed = actual == case.expected
        failure = None if passed else "structured_json output did not exactly match expected JSON value"
        return _result(case, actual=actual, passed=passed, failure=failure)

    raise ValueError(f"unsupported scorer: {case.scorer}")


def score_mtp_pair(
    ordinary: ParsedOutput,
    mtp: ParsedOutput,
    *,
    case: QualityCase | None = None,
    case_id: str | None = None,
    category: str | None = None,
) -> QualityResult:
    mtp_case = _resolve_mtp_case(
        case=case,
        case_id=case_id,
        category=category,
        ordinary=ordinary,
        mtp=mtp,
    )
    if ordinary.token_ids is None or mtp.token_ids is None:
        return _result(
            mtp_case,
            expected=list(ordinary.token_ids) if ordinary.token_ids is not None else None,
            actual=list(mtp.token_ids) if mtp.token_ids is not None else None,
            passed=False,
            failure="ordinary/MTP token ids unavailable for exact comparison",
        )
    actual = list(mtp.token_ids)
    passed = tuple(ordinary.token_ids) == tuple(mtp.token_ids)
    failure = None if passed else "ordinary/MTP token ids did not exactly match"
    return _result(
        mtp_case,
        expected=list(ordinary.token_ids),
        actual=actual,
        passed=passed,
        failure=failure,
    )


def summarize_quality(
    results: tuple[QualityResult, ...] | list[QualityResult],
    *,
    required_cases: tuple[QualityCase, ...] | list[QualityCase] | None = None,
) -> dict[str, object]:
    result_tuple = tuple(results)
    seen_result_ids: set[str] = set()
    for result in result_tuple:
        if result.id in seen_result_ids:
            raise ValueError(f"duplicate quality result id: {result.id}")
        seen_result_ids.add(result.id)

    required_tuple = tuple(required_cases) if required_cases is not None else None
    result_by_id = {result.id: result for result in result_tuple}
    case_entries: list[dict[str, object]] = []
    missing_case_ids: list[str] = []

    if required_tuple is not None:
        required_ids = {case.id for case in required_tuple}
        for case in required_tuple:
            result = result_by_id.get(case.id)
            if result is None:
                missing_case_ids.append(case.id)
                case_entries.append(
                    _result_dict(
                        QualityResult(
                            id=case.id,
                            category=case.category,
                            scorer=case.scorer,
                            passed=False,
                            failure="missing result for required quality case",
                            expected_hash=_hash_value(case.expected),
                            actual_hash=_hash_value(None),
                            expected_value=_display_value(case.expected),
                            actual_value=_display_value(None),
                        )
                    )
                )
                continue
            case_entries.append(_result_dict(result))
        for result in result_tuple:
            if result.id not in required_ids:
                case_entries.append(_result_dict(result))
    else:
        case_entries.extend(_result_dict(result) for result in result_tuple)

    categories: dict[str, dict[str, int]] = {}
    passed = 0
    failed = 0
    for entry in case_entries:
        category = entry["category"]
        bucket = categories.setdefault(category, {"passed": 0, "failed": 0, "total": 0})
        bucket["total"] += 1
        if entry["passed"]:
            bucket["passed"] += 1
            passed += 1
        else:
            bucket["failed"] += 1
            failed += 1

    return {
        "passed": passed,
        "failed": failed,
        "total": len(case_entries),
        "categories": categories,
        "cases": case_entries,
        "missing_case_ids": missing_case_ids,
    }


def _result(
    case: QualityCase,
    *,
    actual: object,
    passed: bool,
    failure: str | None,
    expected: object = _USE_CASE_EXPECTED,
) -> QualityResult:
    compared_expected = case.expected if expected is _USE_CASE_EXPECTED else expected
    return QualityResult(
        id=case.id,
        category=case.category,
        scorer=case.scorer,
        passed=passed,
        failure=failure,
        expected_hash=_hash_value(compared_expected),
        actual_hash=_hash_value(actual),
        expected_value=_display_value(compared_expected),
        actual_value=_display_value(actual),
    )


def _hash_value(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _display_value(value: object) -> str:
    if isinstance(value, str):
        return _bounded_text(value)
    return _bounded_text(canonical_json(value))


def _bounded_text(text: str) -> str:
    if len(text) <= MAX_VALUE_PREVIEW:
        return text
    return f"{text[: MAX_VALUE_PREVIEW - 1]}…"


def _result_dict(result: QualityResult) -> dict[str, object]:
    return {
        "id": result.id,
        "category": result.category,
        "scorer": result.scorer,
        "passed": result.passed,
        "failure": result.failure,
        "expected_hash": result.expected_hash,
        "actual_hash": result.actual_hash,
        "expected_value": result.expected_value,
        "actual_value": result.actual_value,
    }


def _resolve_mtp_case(
    *,
    case: QualityCase | None,
    case_id: str | None,
    category: str | None,
    ordinary: ParsedOutput,
    mtp: ParsedOutput,
) -> QualityCase:
    if case is not None:
        if case_id is not None or category is not None:
            raise ValueError("score_mtp_pair accepts either case or explicit case_id/category, not both")
        return case

    if case_id is None or category is None:
        raise ValueError("score_mtp_pair requires a manifest QualityCase or explicit case_id/category")

    return QualityCase(
        id=case_id,
        category=category,
        prompt="",
        max_new_tokens=max(ordinary.generated_tokens, mtp.generated_tokens, 1),
        scorer="exact_tokens",
        expected=list(ordinary.token_ids) if ordinary.token_ids is not None else None,
        decoding_policy="greedy",
    )
