from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from .adapters import ParsedOutput
from .model import QualityCase, canonical_json, parse_strict_json


MTP_CATEGORY = "ordinary-vs-mtp-token-equality"
MAX_VALUE_PREVIEW = 160
_USE_CASE_EXPECTED = object()
_GOLDENS_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "quality" / "scalar-mtp-goldens-v1.json"
_GOLDEN_ENGINES = {"supersonic-wmma", "supersonic-scalar-lab"}
_GOLDEN_CASES = {
    "ordinary-vs-mtp-token-equality-1",
    "ordinary-vs-mtp-token-equality-2",
}


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


def load_scalar_mtp_goldens(path: Path = _GOLDENS_PATH) -> dict[str, dict[str, tuple[int, ...]]]:
    payload = parse_strict_json(path.read_text(encoding="utf-8"), context="scalar MTP goldens")
    if not isinstance(payload, dict) or set(payload) != {
        "version",
        "artifact",
        "tokenizer_sha256",
        "chat_template_sha256",
        "engines",
    }:
        raise ValueError("scalar MTP goldens must use the closed v1 shape")
    if payload["version"] != "v1" or not isinstance(payload["engines"], dict):
        raise ValueError("scalar MTP goldens must use version v1")
    engines = payload["engines"]
    if set(engines) != _GOLDEN_ENGINES:
        raise ValueError("scalar MTP goldens must contain both SuperSonic engines")
    result: dict[str, dict[str, tuple[int, ...]]] = {}
    for engine_name, engine_value in engines.items():
        if not isinstance(engine_value, dict) or set(engine_value) != {
            "binary_sha256",
            "instruction_stream_sha256",
            "cases",
        }:
            raise ValueError(f"scalar MTP golden engine {engine_name} has an invalid shape")
        cases = engine_value["cases"]
        if not isinstance(cases, dict) or set(cases) != _GOLDEN_CASES:
            raise ValueError(f"scalar MTP golden engine {engine_name} must contain both cases")
        result[engine_name] = {}
        for case_id, case_value in cases.items():
            if not isinstance(case_value, dict) or set(case_value) != {
                "prompt_sha256",
                "token_ids",
                "generated_text",
            }:
                raise ValueError(f"scalar MTP golden {engine_name}/{case_id} has an invalid shape")
            tokens = case_value["token_ids"]
            if (
                not isinstance(tokens, list)
                or not tokens
                or len(tokens) > 8
                or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in tokens)
            ):
                raise ValueError(f"scalar MTP golden {engine_name}/{case_id} has invalid token ids")
            result[engine_name][case_id] = tuple(tokens)
    return result


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
    expected_tokens: tuple[int, ...] | list[int] | None = None,
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
    expected = list(expected_tokens) if expected_tokens is not None else list(ordinary.token_ids)
    actual = list(mtp.token_ids)
    modes_match = tuple(ordinary.token_ids) == tuple(mtp.token_ids)
    golden_matches = list(ordinary.token_ids) == expected
    passed = modes_match and golden_matches
    if not modes_match:
        failure = "ordinary/MTP token ids did not exactly match"
    elif not golden_matches:
        failure = "ordinary/MTP token ids matched each other but not the reviewed golden"
    else:
        failure = None
    return _result(
        mtp_case,
        expected=expected,
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
