from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

from .model import EngineManifest, PerformanceCase


ADAPTER_VERSION = 1
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
LLAMA_CPP_TIMING_PREFIX_RE = re.compile(r"^llama_perf_context_print:\s+")
LLAMA_CPP_PROMPT_EVAL_RE = re.compile(
    r"^llama_perf_context_print:\s+prompt eval time\s+=\s+(?P<ms>[-+A-Za-z0-9.]+)\s+ms\s+/\s+"
    r"(?P<count>[0-9]+)\s+tokens\s+\(\s*(?P<per_tok>[-+A-Za-z0-9.]+)\s+ms per token,\s+"
    r"(?P<tps>[-+A-Za-z0-9.]+)\s+tokens per second\)$",
    re.MULTILINE,
)
LLAMA_CPP_EVAL_RE = re.compile(
    r"^llama_perf_context_print:\s+eval time\s+=\s+(?P<ms>[-+A-Za-z0-9.]+)\s+ms\s+/\s+"
    r"(?P<count>[0-9]+)\s+runs\s+\(\s*(?P<per_tok>[-+A-Za-z0-9.]+)\s+ms per token,\s+"
    r"(?P<tps>[-+A-Za-z0-9.]+)\s+tokens per second\)$",
    re.MULTILINE,
)
LLAMA_CPP_TOTAL_RE = re.compile(
    r"^llama_perf_context_print:\s+total time\s+=\s+(?P<ms>[-+A-Za-z0-9.]+)\s+ms\s+/\s+"
    r"(?P<count>[0-9]+)\s+tokens$",
    re.MULTILINE,
)


@dataclass(frozen=True, slots=True)
class AdapterInputs:
    model_dir: Path
    artifact: Path
    peer_artifact: Path | None = None
    chat: bool = False
    device: int = DEFAULT_DEVICE
    context_size: int | None = None


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


def build_command(
    engine: EngineManifest,
    case: PerformanceCase,
    inputs: AdapterInputs,
) -> tuple[str, ...]:
    _require_engine_scope(engine, case)
    _require_supported_mode(engine, case.mode)

    if engine.name == "supersonic":
        return _build_supersonic_command(engine, case, inputs)
    if engine.name == "llama-cpp":
        return _build_llama_cpp_command(engine, case, inputs)
    raise ValueError(f"unsupported engine adapter: {engine.name}")


def parse_output(engine_name: str, stdout: str, stderr: str = "") -> ParsedOutput:
    if engine_name == "supersonic":
        return _parse_common_output(engine_name, stdout, engine_version=None)
    if engine_name == "llama-cpp":
        from .manifest import load_engine

        combined = _combine_llama_streams(stdout, stderr)
        return _parse_llama_cpp_output(combined, engine_version=load_engine(engine_name).pinned_version)
    raise ValueError(f"unsupported engine adapter: {engine_name}")


def _combine_llama_streams(stdout: str, stderr: str) -> str:
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
        "--ignore-eos",
        "--emit-generated-json",
        "--emit-stage-timings",
        "--device",
        str(_require_non_negative_int(inputs.device, "device")),
    ]
    if inputs.chat:
        args.append("--chat")
    if inputs.context_size is not None:
        args.extend(["--context-size", str(_require_positive_int(inputs.context_size, "context_size"))])
    if case.mode == "mtp":
        args.append("--speculative-decode")
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
        "--model",
        str(inputs.peer_artifact),
        "--prompt",
        case.prompt,
        "--n-predict",
        str(case.max_new_tokens),
        "--ignore-eos",
        "--perf",
        "--show-timings",
        "--no-display-prompt",
        "--temp",
        "0",
        "--top-k",
        "0",
        "--top-p",
        "1",
        "--seed",
        str(DEFAULT_SAMPLING_SEED),
    ]
    if inputs.chat:
        args.extend(["--conversation", "--single-turn"])
    else:
        args.append("--no-conversation")
    if inputs.context_size is not None:
        args.extend(["--ctx-size", str(_require_positive_int(inputs.context_size, "context_size"))])
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
    prompt_match = _extract_match_exactly_one(LLAMA_CPP_PROMPT_EVAL_RE, stdout, "prompt eval time line")
    eval_match = _extract_match_exactly_one(LLAMA_CPP_EVAL_RE, stdout, "eval time line")
    total_match = _extract_match_exactly_one(LLAMA_CPP_TOTAL_RE, stdout, "total time line")

    generated_text = _parse_llama_cpp_generated_text(stdout)
    if not generated_text:
        raise ValueError("llama-cpp output must contain deterministic generated text")

    prompt_tokens = int(prompt_match.group("count"))
    prompt_ms = _require_finite_positive(float(prompt_match.group("ms")), "prompt_eval_ms")
    prompt_ms_per_tok = _require_finite_positive(float(prompt_match.group("per_tok")), "prompt_eval_ms_per_tok")
    prompt_tps = _require_finite_positive(float(prompt_match.group("tps")), "prompt_eval_tokens_per_second")
    _require_rate_consistency(prompt_ms, prompt_tokens, prompt_ms_per_tok, prompt_tps, "prompt eval time")

    generated_tokens = int(eval_match.group("count"))
    decode_ms = _require_finite_positive(float(eval_match.group("ms")), "decode_ms")
    ms_per_tok = _require_finite_positive(float(eval_match.group("per_tok")), "ms_per_tok")
    tokens_per_second = _require_finite_positive(float(eval_match.group("tps")), "tokens_per_second")
    _require_rate_consistency(decode_ms, generated_tokens, ms_per_tok, tokens_per_second, "eval time")

    total_tokens = int(total_match.group("count"))
    total_ms = _require_finite_positive(float(total_match.group("ms")), "total_ms")
    if total_tokens != prompt_tokens + generated_tokens:
        raise ValueError("llama-cpp total time token count is inconsistent with prompt and generated counts")
    if not math.isclose(total_ms, prompt_ms + decode_ms, rel_tol=0.05, abs_tol=2.0):
        raise ValueError("llama-cpp total time is inconsistent with prompt and eval time")

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


def _parse_llama_cpp_generated_text(stdout: str) -> str:
    lines = [line for line in stdout.splitlines() if not LLAMA_CPP_TIMING_PREFIX_RE.match(line)]
    return "\n".join(lines)


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


def _require_finite_positive(value: float, label: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return value


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
