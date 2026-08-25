from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

from .model import EngineManifest, PerformanceCase


ADAPTER_VERSION = 2
MODEL_NAME = "qwen3.8-27b"
DEFAULT_DEVICE = 0
DEFAULT_SAMPLING_SEED = 42
GENERATED_JSON_RE = re.compile(r"^\[generated_json\]\s+(?P<value>.+)$", re.MULTILINE)
TOKENS_RE = re.compile(r"^\[tokens\]\s+(?P<value>.*)$", re.MULTILINE)
RESULT_RE = re.compile(
    r"^\[result\]\s+prompt_tokens=(?P<prompt>[0-9]+)\s+"
    r"generated_tokens=(?P<generated>[0-9]+)\s+"
    r"decode_ms=(?P<decode>[-+A-Za-z0-9.]+)\s+"
    r"ms_per_tok=(?P<per_tok>[-+A-Za-z0-9.]+)\s*$",
    re.MULTILINE,
)
STAGE_TIMINGS_RE = re.compile(
    r"^\[stage-timings\]\s+steps=(?P<steps>[0-9]+)\b.*?\btotal_native_decode_ms=(?P<total>[-+A-Za-z0-9.]+)\b.*$",
    re.MULTILINE,
)
LLAMA_CPP_JSON_RE = re.compile(r"^\[llama_cpp_json\]\s+(?P<value>.+)$", re.MULTILINE)
SUPERSONIC_SCALAR_JSON_RE = re.compile(r"^\[supersonic_json\]\s+(?P<value>.+)$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class AdapterInputs:
    model_dir: Path
    artifact: Path
    peer_artifact: Path | None = None
    chat: bool = False
    device: int = DEFAULT_DEVICE
    context_size: int | None = None
    fixed_token_count: bool = True
    sampling_seed: int = DEFAULT_SAMPLING_SEED


@dataclass(frozen=True, slots=True)
class ParsedOutput:
    engine_name: str
    engine_version: str | None
    generated_text: str
    token_ids: tuple[int, ...] | None
    prompt_tokens: int
    generated_tokens: int
    decode_ms: float
    ms_per_tok: float
    tokens_per_second: float
    lm_head_ms: float | None = None
    timed_decode_steps: int | None = None


def build_command(
    engine: EngineManifest,
    case: PerformanceCase,
    inputs: AdapterInputs,
) -> tuple[str, ...]:
    _require_engine_scope(engine, case)
    _require_supported_mode(engine, case.mode)

    if engine.name in ("supersonic", "supersonic-wmma"):
        return _build_supersonic_command(engine, case, inputs)
    if engine.name == "supersonic-scalar-lab":
        return _build_scalar_lab_command(engine, case, inputs)
    if engine.name == "llama-cpp":
        return _build_llama_cpp_command(engine, case, inputs)
    raise ValueError(f"unsupported engine adapter: {engine.name}")


def parse_output(engine_name: str, stdout: str, stderr: str = "") -> ParsedOutput:
    if engine_name in ("supersonic", "supersonic-wmma"):
        combined = _combine_streams(stdout, stderr)
        return _parse_common_output(engine_name, combined, engine_version=None)
    if engine_name == "supersonic-scalar-lab":
        combined = _combine_streams(stdout, stderr)
        return _parse_scalar_lab_output(combined)
    if engine_name == "llama-cpp":
        from .manifest import load_engine

        combined = _combine_streams(stdout, stderr)
        return _parse_llama_cpp_output(combined, engine_version=load_engine(engine_name).pinned_version)
    raise ValueError(f"unsupported engine adapter: {engine_name}")


def _combine_streams(stdout: str, stderr: str) -> str:
    if not stderr:
        return stdout
    if not stdout:
        return stderr
    return f"{stdout.rstrip(chr(10))}\n{stderr.lstrip(chr(10))}"


def _build_supersonic_command(
    engine: EngineManifest,
    case: PerformanceCase,
    inputs: AdapterInputs,
) -> tuple[str, ...]:
    args = [
        engine.binary,
        "--model",
        MODEL_NAME,
        "--model-dir",
        str(inputs.model_dir),
        "--gguf-file",
        str(inputs.artifact),
        "--prompt",
        case.prompt,
        "--max-new-tokens",
        str(case.max_new_tokens),
        "--emit-generated-json",
        "--emit-stage-timings",
        "--device",
        str(_require_non_negative_int(inputs.device, "device")),
        "--sampling-seed",
        str(_require_non_negative_int(inputs.sampling_seed, "sampling_seed")),
    ]
    if inputs.fixed_token_count:
        args.append("--ignore-eos")
    if inputs.chat:
        args.append("--chat")
    if inputs.context_size is not None:
        args.extend(["--context-size", str(_require_positive_int(inputs.context_size, "context_size"))])
    if case.mode == "mtp":
        args.append("--speculative-decode")
    return tuple(args)


def _build_scalar_lab_command(
    engine: EngineManifest,
    case: PerformanceCase,
    inputs: AdapterInputs,
) -> tuple[str, ...]:
    args = [
        engine.binary,
        "--model-dir",
        str(inputs.model_dir),
        "--artifact",
        str(inputs.artifact),
        "--prompt",
        case.prompt,
        "--max-new-tokens",
        str(case.max_new_tokens),
        "--device",
        str(_require_non_negative_int(inputs.device, "device")),
        "--mode",
        case.mode,
    ]
    if inputs.chat:
        args.append("--chat")
    if not inputs.fixed_token_count:
        args.append("--honor-eos")
    return tuple(args)


def _build_llama_cpp_command(
    engine: EngineManifest,
    case: PerformanceCase,
    inputs: AdapterInputs,
) -> tuple[str, ...]:
    if inputs.peer_artifact is None:
        raise ValueError("llama-cpp requires a separate peer_artifact")

    args = [
        engine.binary,
        "--server-binary",
        "llama-server",
        "--model",
        str(inputs.peer_artifact),
        "--prompt",
        case.prompt,
        "--max-new-tokens",
        str(case.max_new_tokens),
        "--seed",
        str(_require_non_negative_int(inputs.sampling_seed, "sampling_seed")),
    ]
    if inputs.chat:
        args.append("--chat")
    if not inputs.fixed_token_count:
        args.append("--honor-eos")
    if inputs.context_size is not None:
        args.extend(["--context-size", str(_require_positive_int(inputs.context_size, "context_size"))])
    return tuple(args)


def _parse_common_output(
    engine_name: str,
    stdout: str,
    *,
    engine_version: str | None,
) -> ParsedOutput:
    generated_text = _parse_generated_text(stdout)
    token_ids = _parse_token_ids(stdout)
    prompt_tokens, generated_tokens, decode_ms, ms_per_tok = _parse_result(stdout)

    if generated_tokens != len(token_ids):
        raise ValueError(
            f"{engine_name} generated token count {generated_tokens} does not match token ids {len(token_ids)}"
        )
    if prompt_tokens <= 0:
        raise ValueError(f"{engine_name} prompt_tokens must be positive")
    if generated_tokens <= 0:
        raise ValueError(f"{engine_name} generated_tokens must be positive")

    expected_decode_ms = ms_per_tok * generated_tokens
    if not math.isclose(expected_decode_ms, decode_ms, rel_tol=0.05, abs_tol=1.0):
        raise ValueError(
            f"{engine_name} decode_ms {decode_ms} is inconsistent with generated_tokens and ms_per_tok"
        )

    stage = STAGE_TIMINGS_RE.findall(stdout)
    if stage:
        if len(stage) != 1:
            raise ValueError(f"{engine_name} output must contain exactly one stage-timings line")
        stage_steps, stage_total = stage[0]
        if int(stage_steps) != generated_tokens:
            raise ValueError(f"{engine_name} stage-timings steps must match generated_tokens")
        _require_finite_positive(float(stage_total), "total_native_decode_ms")

    tokens_per_second = (generated_tokens * 1000.0) / decode_ms
    _require_finite_positive(tokens_per_second, "tokens_per_second")

    return ParsedOutput(
        engine_name=engine_name,
        engine_version=engine_version,
        generated_text=generated_text,
        token_ids=token_ids,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        decode_ms=decode_ms,
        ms_per_tok=ms_per_tok,
        tokens_per_second=tokens_per_second,
    )


def _parse_llama_cpp_output(stdout: str, *, engine_version: str | None) -> ParsedOutput:
    raw = _extract_exactly_one(LLAMA_CPP_JSON_RE, stdout, "llama_cpp_json line")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("llama_cpp_json must be valid JSON") from exc
    required = {
        "decode_ms",
        "generated_text",
        "generated_tokens",
        "ms_per_tok",
        "prompt_tokens",
        "tokens_per_second",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("llama_cpp_json must contain exactly the normalized peer fields")
    generated_text = payload["generated_text"]
    if not isinstance(generated_text, str) or not generated_text:
        raise ValueError("llama-cpp output must contain deterministic generated text")
    prompt_tokens = _require_positive_int(payload["prompt_tokens"], "prompt_tokens")
    generated_tokens = _require_positive_int(payload["generated_tokens"], "generated_tokens")
    decode_ms = _require_finite_positive(payload["decode_ms"], "decode_ms")
    ms_per_tok = _require_finite_positive(payload["ms_per_tok"], "ms_per_tok")
    tokens_per_second = _require_finite_positive(payload["tokens_per_second"], "tokens_per_second")
    _require_rate_consistency(decode_ms, generated_tokens, ms_per_tok, tokens_per_second, "peer timing")

    return ParsedOutput(
        engine_name="llama-cpp",
        engine_version=engine_version,
        generated_text=generated_text,
        token_ids=None,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        decode_ms=decode_ms,
        ms_per_tok=ms_per_tok,
        tokens_per_second=tokens_per_second,
    )


def _parse_scalar_lab_output(stdout: str) -> ParsedOutput:
    raw = _extract_exactly_one(SUPERSONIC_SCALAR_JSON_RE, stdout, "supersonic_json line")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("supersonic_json must be valid JSON") from exc
    required = {
        "decode_ms",
        "engine_name",
        "engine_version",
        "generated_text",
        "generated_tokens",
        "lm_head_ms",
        "ms_per_tok",
        "prompt_tokens",
        "token_ids",
        "timed_decode_steps",
        "tokens_per_second",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("supersonic_json must contain exactly the normalized scalar fields")
    if payload["engine_name"] != "supersonic-scalar-lab":
        raise ValueError("supersonic_json engine_name must be supersonic-scalar-lab")
    if payload["engine_version"] != "scalar-head-lab-v1":
        raise ValueError("supersonic_json engine_version must be scalar-head-lab-v1")
    generated_text = payload["generated_text"]
    if not isinstance(generated_text, str):
        raise ValueError("supersonic scalar generated_text must be text")
    prompt_tokens = _require_positive_int(payload["prompt_tokens"], "prompt_tokens")
    generated_tokens = _require_positive_int(payload["generated_tokens"], "generated_tokens")
    raw_tokens = payload["token_ids"]
    if (
        not isinstance(raw_tokens, list)
        or len(raw_tokens) != generated_tokens
        or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in raw_tokens)
    ):
        raise ValueError("supersonic scalar token_ids must match generated_tokens")
    decode_ms = _require_finite_positive(payload["decode_ms"], "decode_ms")
    lm_head_ms = _require_finite_positive(payload["lm_head_ms"], "lm_head_ms")
    timed_decode_steps = _require_positive_int(payload["timed_decode_steps"], "timed_decode_steps")
    if timed_decode_steps > generated_tokens:
        raise ValueError("timed_decode_steps cannot exceed generated_tokens")
    ms_per_tok = _require_finite_positive(payload["ms_per_tok"], "ms_per_tok")
    tokens_per_second = _require_finite_positive(payload["tokens_per_second"], "tokens_per_second")
    _require_rate_consistency(
        decode_ms,
        generated_tokens,
        ms_per_tok,
        tokens_per_second,
        "scalar timing",
    )
    return ParsedOutput(
        engine_name="supersonic-scalar-lab",
        engine_version="scalar-head-lab-v1",
        generated_text=generated_text,
        token_ids=tuple(raw_tokens),
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        decode_ms=decode_ms,
        ms_per_tok=ms_per_tok,
        tokens_per_second=tokens_per_second,
        lm_head_ms=lm_head_ms,
        timed_decode_steps=timed_decode_steps,
    )


def _parse_generated_text(stdout: str) -> str:
    raw = _extract_exactly_one(GENERATED_JSON_RE, stdout, "[generated_json] line")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("generated_json must be valid JSON") from exc
    if not isinstance(value, str):
        raise ValueError("generated_json must encode a string")
    return value


def _parse_token_ids(stdout: str) -> tuple[int, ...]:
    raw = _extract_exactly_one(TOKENS_RE, stdout, "[tokens] line")
    if not raw.strip():
        raise ValueError("[tokens] line must list at least one token id")
    values = raw.split()
    try:
        token_ids = tuple(int(value) for value in values)
    except ValueError as exc:
        raise ValueError("token ids must be integers") from exc
    if any(token_id < 0 for token_id in token_ids):
        raise ValueError("token ids must be non-negative")
    return token_ids


def _parse_result(stdout: str) -> tuple[int, int, float, float]:
    match = _extract_match_exactly_one(RESULT_RE, stdout, "[result] line")
    prompt_tokens = int(match.group("prompt"))
    generated_tokens = int(match.group("generated"))
    decode_ms = _require_finite_positive(float(match.group("decode")), "decode_ms")
    ms_per_tok = _require_finite_positive(float(match.group("per_tok")), "ms_per_tok")
    return prompt_tokens, generated_tokens, decode_ms, ms_per_tok


def _extract_exactly_one(pattern: re.Pattern[str], text: str, label: str) -> str:
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise ValueError(f"output must contain exactly one {label}")
    if isinstance(matches[0], tuple):
        raise ValueError(f"internal adapter error extracting {label}")
    return matches[0]


def _extract_match_exactly_one(pattern: re.Pattern[str], text: str, label: str) -> re.Match[str]:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"output must contain exactly one {label}")
    return matches[0]


def _require_engine_scope(engine: EngineManifest, case: PerformanceCase) -> None:
    if engine.name not in case.engines:
        raise ValueError(f"case {case.id!r} does not allow engine {engine.name!r} in its scope")


def _require_supported_mode(engine: EngineManifest, mode: str) -> None:
    if mode not in engine.supported_modes:
        raise ValueError(f"engine {engine.name!r} has unsupported mode {mode!r}")


def _require_positive_int(value: int, label: str) -> int:
    if value <= 0:
        raise ValueError(f"{label} must be positive")
    return value


def _require_non_negative_int(value: int, label: str) -> int:
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _require_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _require_finite_positive(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _require_rate_consistency(
    elapsed_ms: float,
    token_count: int,
    ms_per_tok: float,
    tokens_per_second: float,
    label: str,
) -> None:
    if token_count <= 0:
        raise ValueError(f"{label} token count must be positive")
    if not math.isclose(elapsed_ms, token_count * ms_per_tok, rel_tol=0.05, abs_tol=1.0):
        raise ValueError(f"{label} is inconsistent with token count and ms_per_tok")
    derived_tps = (token_count * 1000.0) / elapsed_ms
    if not math.isclose(tokens_per_second, derived_tps, rel_tol=0.05, abs_tol=1.0):
        raise ValueError(f"{label} is inconsistent with tokens per second")
