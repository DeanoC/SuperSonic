"""Render validated benchmark evidence as a deterministic static site.

The renderer is intentionally a small, dependency-free HTML writer.  Result
records remain the source of truth: validation decides whether a record is
publishable and ``compare_records`` decides whether a peer comparison can
carry a relative result.  This module only formats those decisions.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import html
import json
from pathlib import Path
import re
import shlex
import tempfile

from . import compare, manifest, validation


GENERATOR_VERSION = "1"
SCHEMA_VERSION = "result-v1"
MARKER_NAME = ".benchmark-site"
_METADATA_NAMES = frozenset({"manifest.json", "run-manifest.json", "comparison.json"})
_GENERATED_ROOT_FILES = frozenset({"index.html", "methodology.html"})
_GENERATED_ROOT_DIRS = frozenset({"assets", "runs", "trends", "comparisons"})
_DIMENSIONS: tuple[tuple[str, str], ...] = (
    ("architecture", "hardware.architecture"),
    ("artifact", "artifact.semantic_id"),
    ("workload", "workload.case_id"),
    ("mode", "workload.mode"),
    ("cache-state", "workload.cache_state"),
)
_REASON_FIELDS = {
    "identity": "hardware.identity",
    "identity_kind": "hardware.identity_kind",
    "architecture": "hardware.architecture",
    "physical_gpu": "hardware.physical_gpu",
    "logical_gpu": "hardware.logical_gpu",
    "semantic_id": "artifact.semantic_id",
    "quantization": "artifact.quantization",
    "sha256": "artifact.sha256",
    "tokenizer_sha256": "artifact.tokenizer_sha256",
    "chat_template_sha256": "artifact.chat_template_sha256",
    "case_id": "workload.case_id",
    "prompt_sha256": "workload.prompt_sha256",
    "context_limit": "workload.context_limit",
    "max_new_tokens": "workload.max_new_tokens",
    "mode": "workload.mode",
    "stop_policy": "workload.stop_policy",
    "cache_state": "workload.cache_state",
    "warmups": "workload.warmups",
    "measurement_boundary": "workload.measurement_boundary",
    "clock_policy": "environment.clock_policy",
    "power_cap_watts": "environment.requested.power_cap_watts",
}


def page(title: str, body: str, *, href_prefix: str = "") -> str:
    """Wrap a body in the fixed site document shell.

    The title is escaped here as a final guard even though callers generally
    pass stable text.  The stylesheet path is document-relative at the page's
    depth, so the site works under a project subpath and from local files.
    No page needs JavaScript or an external network dependency.
    """

    stylesheet = f"{href_prefix}assets/benchmarks.css"
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<title>{_escape(title)}</title><link rel=\"stylesheet\" "
        f"href=\"{_escape(stylesheet)}\"></head>"
        f"<body>{body}</body></html>\n"
    )


def render_site(results_root: str | Path, output_root: str | Path) -> tuple[Path, ...]:
    """Render all publishable records below ``results_root``.

    Input records are validated before publication checks.  Valid records that
    are incomplete, dirty, quality-failing, or otherwise not publishable are
    deliberately omitted; malformed records fail closed.  Existing output is
    cleaned only through the renderer-owned manifest, never recursively or by
    touching the results tree.
    """

    results_path = Path(results_root)
    output_path = Path(output_root)
    _check_separate_roots(results_path, output_path)
    records = _load_publishable_records(results_path)
    if not records:
        raise ValueError("no validated publishable benchmark record found")
    owned = _prepare_output(output_path)

    paths: list[Path] = []

    def emit(relative: str, content: str) -> None:
        path = output_path / relative
        _validate_generated_relative(relative)
        _reject_symlink_parents(path, output_path)
        if path.exists() and relative not in owned and relative != MARKER_NAME:
            raise ValueError(f"refusing to overwrite unowned output file: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        paths.append(path)

    run_ids = {id(record): _run_page_id(record) for record in records}
    run_links = {run_ids[id(record)]: f"runs/{run_ids[id(record)]}.html" for record in records}

    emit("index.html", _render_landing(records, run_links))
    emit("methodology.html", _render_methodology())

    for record in records:
        run_id = run_ids[id(record)]
        emit(f"runs/{run_id}.html", _render_run_page(record, href_prefix="../"))

    trend_entries = _trend_entries(records)
    emit("trends/index.html", _render_trend_index(trend_entries, href_prefix="../"))
    for dimension, values in trend_entries:
        for value, dimension_records in values:
            value_id = _value_page_id(dimension, value)
            emit(
                f"trends/{dimension}/{value_id}.html",
                _render_trend_page(
                    dimension,
                    value,
                    dimension_records,
                    run_links,
                    href_prefix="../../",
                ),
            )

    pairs = _comparison_pairs(records)
    comparison_rows: list[tuple[str, dict[str, object], dict[str, object], compare.Comparison]] = []
    for left, right in pairs:
        result = compare.compare_records(left, right)
        comparison_id = _comparison_page_id(left, right)
        comparison_rows.append((comparison_id, left, right, result))
        emit(
            f"comparisons/{comparison_id}.html",
            _render_comparison_page(left, right, result, run_links, href_prefix="../"),
        )
    comparison_rows.sort(key=lambda item: item[0])
    emit("comparisons/index.html", _render_comparison_index(comparison_rows, href_prefix="../"))

    stylesheet = _stylesheet()
    emit("assets/benchmarks.css", stylesheet)

    generated = sorted({path.relative_to(output_path).as_posix() for path in paths})
    marker = {
        "generator_version": GENERATOR_VERSION,
        "files": generated,
    }
    marker_path = output_path / MARKER_NAME
    marker_path.write_text(_canonical_json(marker) + "\n", encoding="utf-8", newline="\n")
    paths.sort(key=lambda path: path.relative_to(output_path).as_posix())
    return tuple(paths)


def render_run(record: Mapping[str, object]) -> str:
    """Render one run page for callers and tests that already hold a record."""

    return _render_run_page(_as_record(record), href_prefix="")


def render_comparison(
    left: Mapping[str, object] | Sequence[Mapping[str, object]],
    right: Mapping[str, object] | None = None,
) -> str:
    """Render one peer comparison using the comparator's decision.

    ``left`` may be a two-record sequence for convenience.  The public site
    always passes two records explicitly, which keeps the evidence links
    stable and makes accidental one-sided comparisons impossible.
    """

    first, second = _comparison_inputs(left, right)
    result = compare.compare_records(first, second)
    return _render_comparison_page(first, second, result, {}, href_prefix="")


def _load_publishable_records(results_path: Path) -> list[dict[str, object]]:
    if not results_path.exists():
        raise ValueError(f"benchmark results path does not exist: {results_path}")
    if results_path.is_file():
        candidates = (results_path,)
    elif results_path.is_dir():
        candidates = tuple(
            sorted(
                path
                for path in results_path.rglob("*.json")
                if path.is_file() and path.name not in _METADATA_NAMES
            )
        )
    else:
        raise ValueError(f"benchmark results path is not a file or directory: {results_path}")

    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for path in candidates:
        value = _load_json(path)
        if not _looks_like_record(value):
            continue
        record = _as_record(value)
        try:
            validation.validate_record(record)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid benchmark record {path}: {exc}") from exc
        try:
            # Publication eligibility remains validator-owned.  The renderer
            # only decides to omit a valid record after this gate rejects it.
            validation._validate_publishable(record, path)  # type: ignore[attr-defined]
        except ValueError:
            continue
        key = _canonical_json(record)
        if key not in seen:
            seen.add(key)
            records.append(record)
    records.sort(key=_record_sort_key)
    return records


def _load_json(path: Path) -> object:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> object:
        raise ValueError(f"non-finite JSON number {token}")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{path} must contain strict JSON: {exc}") from exc


def _looks_like_record(value: object) -> bool:
    return isinstance(value, dict) and all(key in value for key in ("run", "engine", "samples"))


def _as_record(value: Mapping[str, object] | object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("benchmark record must be an object")
    return dict(value)


def _check_separate_roots(results_path: Path, output_path: Path) -> None:
    result_resolved = results_path.resolve()
    output_resolved = output_path.resolve()
    if output_path.is_symlink():
        raise ValueError(f"output root must not be a symlink: {output_path}")
    forbidden_roots = {
        Path("/").resolve(),
        Path.cwd().resolve(),
        Path.home().resolve(),
        Path(tempfile.gettempdir()).resolve(),
        manifest.ROOT.resolve(),
    }
    if output_resolved in forbidden_roots:
        raise ValueError(f"output root is not a dedicated output leaf: {output_path}")
    repo_root = manifest.ROOT.resolve()
    try:
        if repo_root.is_relative_to(output_resolved):
            raise ValueError(f"output root is a repository ancestor, not a dedicated output leaf: {output_path}")
    except AttributeError:  # pragma: no cover - Python 3.8 compatibility
        repo_text = str(repo_root) + "/"
        output_text = str(output_resolved) + "/"
        if repo_text.startswith(output_text):
            raise ValueError(f"output root is a repository ancestor, not a dedicated output leaf: {output_path}")
    temp_root = Path(tempfile.gettempdir()).resolve()
    if output_resolved.parent == temp_root and output_resolved.name.startswith("tmp"):
        raise ValueError(f"output root is a temporary root, not a dedicated output leaf: {output_path}")
    if output_resolved.name.lower() in {
        "tmp",
        "home",
        "var",
        "usr",
        "opt",
        "etc",
        "bin",
        "lib",
        "sbin",
        "benchmarks",
        "results",
        "target",
        "workspaces",
        "worktrees",
        "workspace",
        "repo",
        "repos",
    }:
        raise ValueError(f"output root is not a dedicated output leaf: {output_path}")
    if result_resolved == output_resolved:
        raise ValueError("output root must be separate from benchmark results")
    try:
        if result_resolved.is_relative_to(output_resolved) or output_resolved.is_relative_to(result_resolved):
            raise ValueError("output root must not contain or be contained by benchmark results")
    except AttributeError:  # pragma: no cover - Python 3.8 compatibility
        result_text = str(result_resolved) + "/"
        output_text = str(output_resolved) + "/"
        if result_text.startswith(output_text) or output_text.startswith(result_text):
            raise ValueError("output root must not contain or be contained by benchmark results")


def _prepare_output(output_path: Path) -> set[str]:
    if output_path.is_symlink():
        raise ValueError(f"output root must not be a symlink: {output_path}")
    parent = output_path.parent
    while parent != parent.parent:
        if parent.is_symlink():
            raise ValueError(f"output root parent must not be a symlink: {parent}")
        parent = parent.parent
    if output_path.exists() and not output_path.is_dir():
        raise ValueError(f"output root is not a directory: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    marker_path = output_path / MARKER_NAME
    if marker_path.is_symlink():
        raise ValueError(f"renderer ownership marker must not be a symlink: {marker_path}")
    owned: set[str] = set()
    if not marker_path.exists():
        if any(output_path.iterdir()):
            raise ValueError(f"refusing to delete unowned output files below: {output_path}")
        return owned

    marker = _load_json(marker_path)
    if not isinstance(marker, dict) or marker.get("generator_version") != GENERATOR_VERSION:
        raise ValueError(f"invalid renderer ownership marker: {marker_path}")
    files = marker.get("files")
    if not isinstance(files, list) or any(not isinstance(item, str) for item in files):
        raise ValueError(f"invalid renderer ownership file list: {marker_path}")
    for relative in files:
        _validate_generated_relative(relative)
        if relative in owned:
            raise ValueError(f"duplicate renderer ownership entry: {relative}")
        owned.add(relative)
    for directory_name in sorted(_GENERATED_ROOT_DIRS):
        directory = output_path / directory_name
        if directory.is_symlink():
            raise ValueError(f"renderer output directory must not be a symlink: {directory}")
    targets: list[Path] = []
    for relative in sorted(owned):
        target = output_path / relative
        _reject_symlink_parents(target, output_path)
        if target.exists() and not target.is_file():
            raise ValueError(f"renderer ownership target is not a file: {target}")
        targets.append(target)
    for target in targets:
        if target.exists():
            target.unlink()
    # Remove only empty directories that the renderer owns.  User-created
    # files inside one of these directories keep the directory intact.
    for directory_name in sorted(_GENERATED_ROOT_DIRS):
        directory = output_path / directory_name
        if directory.is_dir():
            for child in sorted(directory.rglob("*"), reverse=True):
                if child.is_symlink():
                    continue
                if child.is_dir():
                    try:
                        child.rmdir()
                    except OSError:
                        pass
            try:
                directory.rmdir()
            except OSError:
                pass
    return owned


def _reject_symlink_parents(path: Path, output_path: Path) -> None:
    parent = path.parent
    while True:
        if parent.is_symlink():
            raise ValueError(f"generated output path traverses a symlink: {parent}")
        if parent == output_path or parent == parent.parent:
            return
        parent = parent.parent


def _validate_generated_relative(relative: str) -> None:
    if not relative or relative.startswith(("/", "\\")):
        raise ValueError(f"unsafe generated output path: {relative!r}")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or any(part == "" for part in path.parts):
        raise ValueError(f"unsafe generated output path: {relative!r}")
    if relative == MARKER_NAME:
        raise ValueError(f"renderer marker cannot own itself: {relative!r}")
    if path.parts[0] not in _GENERATED_ROOT_DIRS and relative not in _GENERATED_ROOT_FILES:
        raise ValueError(f"unexpected generated output path: {relative!r}")


def _record_sort_key(record: Mapping[str, object]) -> tuple[str, ...]:
    run = _mapping(record, "run")
    environment = _mapping(record, "environment")
    return (
        str(environment.get("requested_at", "")),
        str(run.get("commit", "")),
        str(run.get("run_id", "")),
        str(_value(record, "engine.name", "")),
        str(_value(record, "workload.case_id", "")),
        _canonical_json(record),
    )


def _run_page_id(record: Mapping[str, object]) -> str:
    run = _value(record, "run.run_id", "run")
    return f"{_slug(str(run))}-{_stable_id(record)[:16]}"


def _value_page_id(dimension: str, value: object) -> str:
    return f"{_slug(str(value))}-{_stable_id((dimension, value))[:16]}"


def _comparison_page_id(left: Mapping[str, object], right: Mapping[str, object]) -> str:
    values = sorted((_canonical_json(left), _canonical_json(right)))
    return f"{_stable_id(values)[:24]}"


def _stable_id(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _slug(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "value"


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _escape(value: object) -> str:
    if value is None:
        return "—"
    return html.escape(str(value), quote=True)


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _value(record: Mapping[str, object], path: str, default: object = None) -> object:
    current: object = record
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _kv_rows(rows: Iterable[tuple[str, object]]) -> str:
    return "".join(f"<dt>{_escape(label)}</dt><dd>{_escape(value)}</dd>" for label, value in rows)


def _navigation(href_prefix: str) -> str:
    home = f"{href_prefix}index.html"
    methodology = f"{href_prefix}methodology.html"
    trends = f"{href_prefix}trends/index.html"
    comparisons = f"{href_prefix}comparisons/index.html"
    return (
        '<nav aria-label="Primary">'
        f'<a href="{_escape(home)}">Benchmark home</a>'
        f'<a href="{_escape(methodology)}">Methodology</a>'
        f'<a href="{_escape(trends)}">Trends</a>'
        f'<a href="{_escape(comparisons)}">Comparisons</a>'
        "</nav>"
    )


def _badge(label: str, value: object, css: str = "") -> str:
    class_name = f"badge {css}".strip()
    return f'<span class="{_escape(class_name)}">{_escape(label)}: {_escape(value)}</span>'


def _run_badges(record: Mapping[str, object]) -> str:
    return " ".join(
        (
            _badge("clock", _value(record, "environment.clock_policy"), "clock"),
            _badge("cache", _value(record, "workload.cache_state"), "cache"),
            _badge("quality", f"{_value(record, 'quality.passed', 0)}/{_value(record, 'quality.total', 0)}", "quality"),
            _badge("status", _value(record, "status.state"), "status"),
        )
    )


def _evidence_block(record: Mapping[str, object], *, href_prefix: str = "") -> str:
    hardware = _mapping(_value(record, "hardware", {}), "hardware")
    artifact = _mapping(_value(record, "artifact", {}), "artifact")
    workload = _mapping(_value(record, "workload", {}), "workload")
    run = _mapping(_value(record, "run", {}), "run")
    environment = _mapping(_value(record, "environment", {}), "environment")
    requested = _mapping(environment.get("requested", {}), "environment.requested")
    quality = _mapping(_value(record, "quality", {}), "quality")
    identity_fields = _mapping(hardware.get("identity_fields", {}), "hardware.identity_fields")
    rows = [
        ("run id", run.get("run_id")),
        ("commit", run.get("commit")),
        ("dirty tree", run.get("dirty")),
        ("result schema version", run.get("schema_version")),
        ("suite / suite version", f"{run.get('suite')} / {run.get('suite_version')}"),
        ("quality version", run.get("quality_version")),
        ("engine", _value(record, "engine.name")),
        ("engine version", _value(record, "engine.version")),
        ("adapter version", _value(record, "engine.adapter_version")),
        ("ROCm version", environment.get("rocm_version")),
        ("HIP version", environment.get("hip_version")),
        ("GPU market name", identity_fields.get("market_name")),
        ("GPU identity", hardware.get("identity")),
        ("GPU identity source SHA-256", hardware.get("identity_source_sha256")),
        ("GPU architecture", hardware.get("architecture")),
        ("physical GPU", hardware.get("physical_gpu")),
        ("logical GPU", hardware.get("logical_gpu")),
        ("artifact semantic id", artifact.get("semantic_id")),
        ("artifact quantization", artifact.get("quantization")),
        ("artifact SHA-256", artifact.get("sha256")),
        ("tokenizer SHA-256", artifact.get("tokenizer_sha256")),
        ("chat template SHA-256", artifact.get("chat_template_sha256")),
        ("workload case", workload.get("case_id")),
        ("prompt SHA-256", workload.get("prompt_sha256")),
        ("context limit", workload.get("context_limit")),
        ("maximum new tokens", workload.get("max_new_tokens")),
        ("generation mode", workload.get("mode")),
        ("stop policy", workload.get("stop_policy")),
        ("measurement boundary", workload.get("measurement_boundary")),
        ("warmups", workload.get("warmups")),
        ("correctness", f"{quality.get('passed', 0)}/{quality.get('total', 0)} passed"),
        ("quality failures", quality.get("failed")),
        ("clock policy", environment.get("clock_policy")),
        ("requested GPU clock MHz", requested.get("gpu_clock_mhz")),
        ("accepted GPU clock tolerance MHz", requested.get("clock_tolerance_mhz")),
        ("requested memory clock MHz", requested.get("memory_clock_mhz")),
        ("requested power cap watts", requested.get("power_cap_watts")),
        ("requested performance level", requested.get("performance_level")),
        ("requested at", environment.get("requested_at")),
        ("observed before at", environment.get("observed_before_at")),
        ("observed after at", environment.get("observed_after_at")),
        ("CPU governor", environment.get("cpu_governor")),
        ("cache state", environment.get("cache_state")),
        ("cache evidence", _canonical_json(environment.get("cache_evidence", {}))),
        ("process reuse", environment.get("process_reuse")),
        ("headline eligible", environment.get("headline_eligible")),
    ]
    allowlisted = environment.get("allowlisted_environment", {})
    if isinstance(allowlisted, Mapping):
        for key in sorted(allowlisted):
            rows.append((f"environment {key}", allowlisted[key]))
    errors = environment.get("verification_errors", [])
    if isinstance(errors, Sequence) and not isinstance(errors, (str, bytes)):
        rows.append(("clock verification errors", "; ".join(str(item) for item in errors) or "none"))
    command = run.get("command", [])
    if isinstance(command, Sequence) and not isinstance(command, (str, bytes)):
        try:
            command_text = shlex.join(str(item) for item in command)
        except (TypeError, ValueError):
            command_text = " ".join(str(item) for item in command)
    else:
        command_text = str(command)
    rows.append(("reproduction command", command_text))
    return (
        '<section class="evidence" id="evidence"><h2>Evidence</h2>'
        '<p>Every reported metric below is tied to this commit, GPU, artifact, '
        'workload, sample set, correctness result, clock policy, and cache state.</p>'
        f"<dl>{_kv_rows(rows)}</dl></section>"
    )


def _sample_values(record: Mapping[str, object], field: str) -> list[float]:
    samples = _value(record, "samples", [])
    values: list[float] = []
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        return values
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        value = sample.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        values.append(float(value))
    return values


def _summary_rows(values: Sequence[float]) -> str:
    if not values:
        return '<p class="empty">No valid samples.</p>'
    summary = compare.summarize_samples(tuple(values))
    rows = (
        ("count", summary.count),
        ("minimum", summary.minimum),
        ("median", summary.median),
        ("maximum", summary.maximum),
        ("median absolute deviation", summary.mad),
    )
    return f"<dl class=\"metrics\">{_kv_rows(rows)}</dl>"


def _samples_block(record: Mapping[str, object]) -> str:
    samples = _value(record, "samples", [])
    rows: list[str] = []
    if isinstance(samples, Sequence) and not isinstance(samples, (str, bytes)):
        for index, sample in enumerate(samples, start=1):
            if isinstance(sample, Mapping):
                decode = sample.get("decode_ms")
                throughput = sample.get("tokens_per_second")
            else:
                decode = throughput = None
            rows.append(
                f"<tr><td>{index}</td><td>{_escape(_fmt(decode))}</td>"
                f"<td>{_escape(_fmt(throughput))}</td><td><a href=\"#evidence\">run evidence</a></td></tr>"
            )
    decode_values = _sample_values(record, "decode_ms")
    tps_values = _sample_values(record, "tokens_per_second")
    return (
        '<section id="performance"><h2>Performance samples</h2>'
        '<p>Raw samples are retained in measured order. Statistics use median, '
        'minimum, maximum, and median absolute deviation.</p>'
        '<table><caption>Raw measured decode samples linked to run evidence</caption>'
        '<thead><tr><th>Sample</th><th>Decode ms</th><th>Tokens/s</th><th>Evidence</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
        '<div class="metric-columns"><div><h3>Decode milliseconds</h3>'
        f"{_summary_rows(decode_values)}</div><div><h3>Tokens per second</h3>{_summary_rows(tps_values)}</div></div>"
        '</section>'
    )


def _telemetry_block(record: Mapping[str, object]) -> str:
    environment = _mapping(_value(record, "environment", {}), "environment")
    entries: list[tuple[str, Mapping[str, object]]] = []
    for label, value in (
        ("before", environment.get("observed_before")),
        ("sample", None),
        ("after", environment.get("observed_after")),
    ):
        if isinstance(value, Mapping):
            entries.append((label, value))
    telemetry = environment.get("telemetry_samples", [])
    if isinstance(telemetry, Sequence) and not isinstance(telemetry, (str, bytes)):
        for index, value in enumerate(telemetry, start=1):
            if isinstance(value, Mapping):
                entries.insert(-1 if entries and entries[-1][0] == "after" else len(entries), (f"sample {index}", value))
    rows: list[str] = []
    for label, value in entries:
        cells = "".join(
            f"<td>{_escape(_fmt(value.get(key)))}</td>"
            for key in (
                "gpu_clock_mhz",
                "memory_clock_mhz",
                "power_cap_watts",
                "power_watts",
                "temperature_celsius",
                "gpu_utilization_percent",
                "memory_utilization_percent",
                "performance_level",
            )
        )
        rows.append(f"<tr><th>{_escape(label)}</th>{cells}<td><a href=\"#evidence\">run evidence</a></td></tr>")
    return (
        '<section id="telemetry"><h2>Clock, power, and thermal evidence</h2>'
        '<table><caption>Observed telemetry linked to commit and clock policy evidence</caption>'
        '<thead><tr><th>Point</th><th>GPU MHz</th><th>Memory MHz</th><th>Power cap W</th>'
        '<th>Power W</th><th>Temperature °C</th><th>GPU util %</th><th>Memory util %</th>'
        '<th>Performance level</th><th>Evidence</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></section>"
    )


def _quality_block(record: Mapping[str, object]) -> str:
    quality = _mapping(_value(record, "quality", {}), "quality")
    categories = quality.get("categories", {})
    category_rows: list[str] = []
    if isinstance(categories, Mapping):
        for name in sorted(categories):
            values = categories[name]
            values = values if isinstance(values, Mapping) else {}
            category_rows.append(
                f"<tr><th>{_escape(name)}</th><td>{_escape(_fmt(values.get('passed')))}</td>"
                f"<td>{_escape(_fmt(values.get('failed')))}</td><td>{_escape(_fmt(values.get('total')))}</td></tr>"
            )
    case_rows: list[str] = []
    cases = quality.get("cases", [])
    if isinstance(cases, Sequence) and not isinstance(cases, (str, bytes)):
        sorted_cases = sorted(
            (case for case in cases if isinstance(case, Mapping)),
            key=lambda case: str(case.get("id", "")),
        )
        for case in sorted_cases:
            failure = case.get("failure")
            outcome = "passed" if case.get("passed") is True else "failed"
            case_rows.append(
                f"<tr><th>{_escape(case.get('id'))}</th><td>{_escape(case.get('category'))}</td>"
                f"<td>{_escape(case.get('scorer'))}</td><td>{_escape(outcome)}</td>"
                f"<td>{_escape(failure if failure is not None else '—')}</td>"
                f"<td>{_escape(case.get('expected_hash'))}</td><td>{_escape(case.get('actual_hash'))}</td>"
                f"<td>{_escape(case.get('expected_value'))}</td><td>{_escape(case.get('actual_value'))}</td></tr>"
            )
    missing = quality.get("missing_case_ids", [])
    missing_text = ", ".join(sorted(str(item) for item in missing)) if isinstance(missing, Sequence) else str(missing)
    return (
        '<section id="quality"><h2>Deterministic quality</h2>'
        f"<p>Aggregate: {_escape(_fmt(quality.get('passed')))} passed, "
        f"{_escape(_fmt(quality.get('failed')))} failed, {_escape(_fmt(quality.get('total')))} total. "
        f"Missing cases: {_escape(missing_text or 'none')}.</p>"
        '<table><caption>Quality category counts</caption><thead><tr><th>Category</th>'
        '<th>Passed</th><th>Failed</th><th>Total</th></tr></thead>'
        f"<tbody>{''.join(category_rows)}</tbody></table>"
        '<table><caption>Quality cases and failure evidence</caption><thead><tr><th>Case</th>'
        '<th>Category</th><th>Scorer</th><th>Outcome</th><th>Failure</th><th>Expected hash</th>'
        '<th>Actual hash</th><th>Expected value</th><th>Actual value</th></tr></thead>'
        f"<tbody>{''.join(case_rows)}</tbody></table></section>"
    )


def _render_run_page(record: Mapping[str, object], *, href_prefix: str) -> str:
    run_id = _value(record, "run.run_id", "run")
    command = _value(record, "run.command", [])
    body = (
        _navigation(href_prefix)
        + '<main class="container">'
        + f"<h1>Benchmark run: {_escape(run_id)}</h1>"
        + f'<p class="badges">{_run_badges(record)}</p>'
        + _evidence_block(record, href_prefix=href_prefix)
        + _samples_block(record)
        + _telemetry_block(record)
        + _quality_block(record)
        + '<section id="reproduce"><h2>Reproduce this run</h2>'
        + '<p>Check out the recorded commit, select the recorded GPU and artifact, '
        + 'then run the command shown in the evidence block. The command is displayed '
        + 'as recorded and is not executed by the site.</p>'
        + f"<pre><code>{_escape(shlex.join(str(item) for item in command) if isinstance(command, Sequence) and not isinstance(command, (str, bytes)) else command)}</code></pre>"
        + "</section></main>"
    )
    return page(f"Benchmark run: {run_id}", body, href_prefix=href_prefix)


def _render_landing(records: Sequence[Mapping[str, object]], run_links: Mapping[str, str]) -> str:
    qualified = [record for record in records if _value(record, "engine.name") == "supersonic"]
    latest = max(qualified, key=_record_sort_key) if qualified else None
    rows: list[str] = []
    for record in sorted(records, key=_record_sort_key, reverse=True):
        run_id = _run_page_id(record)
        stats = _sample_summary(record)
        href = run_links.get(run_id, f"runs/{run_id}.html")
        rows.append(_trend_row(record, stats, href))
    latest_html = "<section class=\"headline\"><h2>No qualified SuperSonic result</h2>"
    latest_html += (
        "<p>No qualified SuperSonic record is available; published peer records remain visible below as context.</p></section>"
        if records
        else "<p>No publishable benchmark records are available.</p></section>"
    )
    if latest is not None:
        run_id = _run_page_id(latest)
        href = run_links.get(run_id, f"runs/{run_id}.html")
        stats = _sample_summary(latest)
        latest_html = (
            '<section class="headline"><h2>Latest qualified SuperSonic result</h2>'
            f"<p><a href=\"{_escape(href)}#evidence\">{_escape(_value(latest, 'run.run_id'))}</a> "
            f"on {_escape(_value(latest, 'hardware.architecture'))} / {_escape(_value(latest, 'hardware.identity'))}, "
            f"artifact {_escape(_value(latest, 'artifact.semantic_id'))}, workload "
            f"{_escape(_value(latest, 'workload.case_id'))}, clock {_escape(_value(latest, 'environment.clock_policy'))}, "
            f"cache {_escape(_value(latest, 'workload.cache_state'))}.</p>"
            f"<p class=\"metric\"><a href=\"{_escape(href)}#performance\">Median decode: {_escape(_fmt(stats.median if stats else None))} ms</a> "
            f"from {_escape(_fmt(stats.count if stats else 0))} raw samples; correctness "
            f"{_escape(_value(latest, 'quality.passed', 0))}/{_escape(_value(latest, 'quality.total', 0))} passed.</p></section>"
        )
    body = (
        _navigation("")
        + '<main class="container"><h1>Reproducible benchmark results</h1>'
        + '<p>This static site is rebuilt from validated, publishable result records. '
        + 'Generated HTML is disposable; records and manifests remain the source of truth.</p>'
        + latest_html
        + '<section><h2>Published runs</h2>'
        + '<table><caption>Each metric links to the run evidence page</caption><thead><tr>'
        + _trend_headers()
        + f"</tr></thead><tbody>{''.join(rows)}</tbody></table></section></main>"
    )
    return page("Reproducible benchmark results", body)


def _sample_summary(record: Mapping[str, object]) -> compare.SampleSummary | None:
    values = _sample_values(record, "decode_ms")
    if not values:
        return None
    return compare.summarize_samples(values)


def _trend_headers() -> str:
    return (
        "<th>Run</th><th>Engine / version</th><th>Median decode ms</th><th>Minimum decode ms</th>"
        "<th>Maximum decode ms</th><th>MAD decode ms</th><th>Samples</th>"
        "<th>Commit</th><th>GPU</th><th>Artifact</th><th>Workload</th>"
        "<th>Correctness</th><th>Clock</th><th>Cache</th>"
    )


def _trend_row(record: Mapping[str, object], stats: compare.SampleSummary | None, href: str) -> str:
    run_id = _value(record, "run.run_id")
    evidence_href = f"{href}#evidence"
    performance_href = f"{href}#performance"
    median = stats.median if stats else None
    minimum = stats.minimum if stats else None
    maximum = stats.maximum if stats else None
    mad = stats.mad if stats else None
    count = stats.count if stats else 0
    return (
        f"<tr><th><a href=\"{_escape(evidence_href)}\">{_escape(run_id)}</a></th>"
        f"<td>{_escape(_value(record, 'engine.name'))} / {_escape(_value(record, 'engine.version'))}</td>"
        f"<td><a href=\"{_escape(performance_href)}\">{_escape(_fmt(median))}</a></td>"
        f"<td>{_escape(_fmt(minimum))}</td><td>{_escape(_fmt(maximum))}</td>"
        f"<td>{_escape(_fmt(mad))}</td>"
        f"<td>{_escape(_fmt(count))}</td>"
        f"<td>{_escape(_value(record, 'run.commit'))}</td>"
        f"<td>{_escape(_value(record, 'hardware.architecture'))} / {_escape(_value(record, 'hardware.identity'))}</td>"
        f"<td>{_escape(_value(record, 'artifact.semantic_id'))} / {_escape(_value(record, 'artifact.sha256'))}</td>"
        f"<td>{_escape(_value(record, 'workload.case_id'))} / {_escape(_value(record, 'workload.prompt_sha256'))}</td>"
        f"<td>{_escape(_value(record, 'quality.passed', 0))}/{_escape(_value(record, 'quality.total', 0))}</td>"
        f"<td>{_escape(_value(record, 'environment.clock_policy'))}</td>"
        f"<td>{_escape(_value(record, 'workload.cache_state'))}</td></tr>"
    )


def _trend_entries(records: Sequence[Mapping[str, object]]) -> tuple[tuple[str, tuple[tuple[str, tuple[Mapping[str, object], ...]], ...]], ...]:
    output: list[tuple[str, tuple[tuple[str, tuple[Mapping[str, object], ...]], ...]]] = []
    for dimension, field in _DIMENSIONS:
        groups: dict[str, list[Mapping[str, object]]] = {}
        for record in records:
            value = str(_value(record, field, "unknown"))
            groups.setdefault(value, []).append(record)
        values = tuple(
            (value, tuple(sorted(group, key=_record_sort_key)))
            for value, group in sorted(groups.items(), key=lambda item: item[0])
        )
        output.append((dimension, values))
    return tuple(output)


def _render_trend_index(
    entries: Sequence[tuple[str, Sequence[tuple[str, Sequence[Mapping[str, object]]]]]],
    *,
    href_prefix: str,
) -> str:
    sections: list[str] = []
    for dimension, values in entries:
        links = "".join(
            f"<li><a href=\"{_escape(dimension)}/{_escape(_value_page_id(dimension, value))}.html\">"
            f"{_escape(value)}</a> ({_escape(_fmt(len(records)))})</li>"
            for value, records in values
        )
        sections.append(f"<section><h2>{_escape(_dimension_label(dimension))}</h2><ul>{links}</ul></section>")
    body = (
        _navigation(href_prefix)
        + '<main class="container"><h1>Benchmark trends</h1>'
        + '<p>Trend pages are separated by architecture, artifact, workload, generation mode, and cache state. '
        + 'Each metric links to the complete run evidence.</p>'
        + "".join(sections)
        + "</main>"
    )
    return page("Benchmark trends", body, href_prefix=href_prefix)


def _render_trend_page(
    dimension: str,
    value: str,
    records: Sequence[Mapping[str, object]],
    run_links: Mapping[str, str],
    *,
    href_prefix: str,
) -> str:
    rows = []
    for record in sorted(records, key=_record_sort_key):
        run_id = _run_page_id(record)
        href = f"{href_prefix}{run_links.get(run_id, f'runs/{run_id}.html')}"
        rows.append(_trend_row(record, _sample_summary(record), href))
    body = (
        _navigation(href_prefix)
        + '<main class="container">'
        + f"<h1>Trend: {_escape(_dimension_label(dimension))} = {_escape(value)}</h1>"
        + '<p>Numbers are shown with commit, GPU, artifact, workload, correctness, clock, cache, and raw-sample links.</p>'
        + '<table><caption>Deterministically sorted trend records</caption><thead><tr>'
        + _trend_headers()
        + f"</tr></thead><tbody>{''.join(rows)}</tbody></table></main>"
    )
    return page(f"Trend: {_dimension_label(dimension)} = {value}", body, href_prefix=href_prefix)


def _dimension_label(dimension: str) -> str:
    return {
        "cache-state": "cache state",
        "mode": "generation mode",
    }.get(dimension, dimension)


def _comparison_inputs(
    left: Mapping[str, object] | Sequence[Mapping[str, object]],
    right: Mapping[str, object] | None,
) -> tuple[dict[str, object], dict[str, object]]:
    if right is None:
        if isinstance(left, Sequence) and not isinstance(left, (str, bytes)) and len(left) == 2:
            first, second = left[0], left[1]
            if isinstance(first, Mapping) and isinstance(second, Mapping):
                return _as_record(first), _as_record(second)
        if isinstance(left, Mapping) and isinstance(left.get("left"), Mapping) and isinstance(left.get("right"), Mapping):
            return _as_record(left["left"]), _as_record(left["right"])
        raise TypeError("render_comparison requires two benchmark records")
    return _as_record(left), _as_record(right)


def _comparison_pairs(records: Sequence[Mapping[str, object]]) -> tuple[tuple[dict[str, object], dict[str, object]], ...]:
    groups: dict[str, list[dict[str, object]]] = {}
    for record in records:
        groups.setdefault(str(_value(record, "run.suite", "")), []).append(_as_record(record))
    pairs: list[tuple[dict[str, object], dict[str, object]]] = []
    seen: set[tuple[str, str]] = set()
    for group in groups.values():
        ordered = sorted(group, key=_record_sort_key)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                if _value(left, "engine.name") == _value(right, "engine.name"):
                    continue
                identity = tuple(sorted((_canonical_json(left), _canonical_json(right))))
                if identity in seen:
                    continue
                seen.add(identity)
                pairs.append((left, right))
    pairs.sort(key=lambda pair: _comparison_page_id(*pair))
    return tuple(pairs)


def _render_comparison_page(
    left: Mapping[str, object],
    right: Mapping[str, object],
    result: compare.Comparison,
    run_links: Mapping[str, str],
    *,
    href_prefix: str,
) -> str:
    left_id = _run_page_id(left)
    right_id = _run_page_id(right)
    left_href = f"{href_prefix}{run_links.get(left_id, f'runs/{left_id}.html')}"
    right_href = f"{href_prefix}{run_links.get(right_id, f'runs/{right_id}.html')}"
    left_stats = result.left
    right_stats = result.right
    rows = (
        _comparison_metric_row("left", left, left_stats, left_href),
        _comparison_metric_row("right", right, right_stats, right_href),
    )
    reasons = "".join(f"<li>{_escape(reason)}</li>" for reason in result.reasons)
    if result.comparable:
        relative = (
            '<section id="relative"><h2>Comparable speedup</h2>'
            f"<p class=\"metric\"><a href=\"{_escape(left_href)}#evidence\">{_escape(_value(left, 'engine.name'))}</a> "
            f"relative to <a href=\"{_escape(right_href)}#evidence\">{_escape(_value(right, 'engine.name'))}</a>: "
            f"{_escape(_fmt(result.speedup))}x using the comparator's median decode samples.</p>"
            '<p>The relative value is published only because the validator marked all '
            'hardware, artifact, workload, clock, cache, correctness, and timing fields comparable.</p></section>'
        )
        status = '<span class="badge comparable">comparable</span>'
    else:
        relative = ""
        status = '<span class="badge qualified">not comparable</span>'
    reason_block = ""
    if not result.comparable:
        reason_block = (
            '<section id="reasons"><h2>Validator reasons</h2>'
            '<p>The peer remains visible as qualified context. The validator supplied these reasons; '
            'the renderer does not relax them.</p>'
            f"<ul>{reasons or '<li>comparison rejected without a reason</li>'}</ul></section>"
        )
    body = (
        _navigation(href_prefix)
        + '<main class="container">'
        + f"<h1>Peer comparison: {_escape(_value(left, 'engine.name'))} vs {_escape(_value(right, 'engine.name'))}</h1>"
        + f"<p class=\"badges\">{status}</p>"
        + relative
        + reason_block
        + '<section id="comparison-evidence"><h2>Comparison evidence</h2>'
        + '<p>Both rows retain the commit, GPU identity, artifact digests, workload, correctness, clock, cache, and raw sample links.</p>'
        + '<table><caption>Median and distribution evidence</caption><thead><tr><th>Side</th><th>Engine / version</th>'
        + '<th>Median decode ms</th><th>Minimum</th><th>Maximum</th><th>MAD</th><th>Samples</th><th>Evidence</th></tr></thead>'
        + f"<tbody>{''.join(rows)}</tbody></table></section>"
        + _comparison_field_evidence(left, right)
        + "</main>"
    )
    return page(
        f"Peer comparison: {_value(left, 'engine.name')} vs {_value(right, 'engine.name')}",
        body,
        href_prefix=href_prefix,
    )


def _comparison_metric_row(
    side: str,
    record: Mapping[str, object],
    summary: compare.SampleSummary,
    href: str,
) -> str:
    return (
        f"<tr><th>{_escape(side)}</th><td>{_escape(_value(record, 'engine.name'))} / {_escape(_value(record, 'engine.version'))}</td>"
        f"<td><a href=\"{_escape(href)}#performance\">{_escape(_fmt(summary.median))}</a></td>"
        f"<td>{_escape(_fmt(summary.minimum))}</td><td>{_escape(_fmt(summary.maximum))}</td>"
        f"<td>{_escape(_fmt(summary.mad))}</td><td>{_escape(_fmt(summary.count))}</td>"
        f"<td><a href=\"{_escape(href)}#evidence\">commit/GPU/artifact/workload/quality/clock/cache</a></td></tr>"
    )


def _comparison_field_evidence(left: Mapping[str, object], right: Mapping[str, object]) -> str:
    rows: list[str] = []
    for label, path in _REASON_FIELDS.items():
        left_value = _value(left, path)
        right_value = _value(right, path)
        rows.append(
            f"<tr><th>{_escape(label)}</th><td>{_escape(left_value)}</td><td>{_escape(right_value)}</td></tr>"
        )
    return (
        '<section id="field-evidence"><h2>Field identity</h2>'
        '<table><caption>Fields supplied to compare_records</caption><thead><tr><th>Field</th><th>Left</th><th>Right</th></tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></section>"
    )


def _render_comparison_index(
    rows: Sequence[tuple[str, Mapping[str, object], Mapping[str, object], compare.Comparison]],
    *,
    href_prefix: str,
) -> str:
    table_rows: list[str] = []
    for comparison_id, left, right, result in rows:
        status = "comparable" if result.comparable else "not comparable"
        reasons = ", ".join(result.reasons) if result.reasons else "—"
        href = f"{comparison_id}.html"
        table_rows.append(
            f"<tr><th><a href=\"{_escape(href)}\">{_escape(_value(left, 'engine.name'))} vs "
            f"{_escape(_value(right, 'engine.name'))}</a></th><td>{_escape(status)}</td>"
            f"<td>{_escape(reasons)}</td><td>{_escape(_value(left, 'workload.case_id'))}</td></tr>"
        )
    body = (
        _navigation(href_prefix)
        + '<main class="container"><h1>Peer comparisons</h1>'
        + '<p>Comparability and reasons come directly from <code>compare_records</code>. '
        + 'Unlike artifacts, clocks, cache states, workloads, or hardware are never presented as a relative result.</p>'
        + '<table><caption>Deterministically sorted comparison decisions</caption><thead><tr>'
        + '<th>Pair</th><th>Status</th><th>Validator reasons</th><th>Workload</th></tr></thead>'
        + f"<tbody>{''.join(table_rows)}</tbody></table></main>"
    )
    return page("Peer comparisons", body, href_prefix=href_prefix)


def _render_methodology() -> str:
    suite_rows: list[str] = []
    for name in ("quick", "full"):
        suite = manifest.load_suite(name)
        suite_rows.append(
            f"<tr><th>{_escape(suite.name)}</th><td>{_escape(_fmt(suite.version))}</td>"
            f"<td>{_escape(_fmt(suite.budget_seconds))}</td><td>{_escape(suite.quality_version)}</td>"
            f"<td>{_escape(_fmt(len(suite.performance_cases)))}</td><td>{_escape(', '.join(suite.engines))}</td></tr>"
        )
    body = (
        _navigation("")
        + '<main class="container"><h1>Benchmark methodology</h1>'
        + '<p>This page is generated from the versioned suite manifests and result schema. '
        + 'The renderer has no JavaScript and performs no network requests.</p>'
        + '<section><h2>Versioned inputs</h2><dl>'
        + f"{_kv_rows((('generator version', GENERATOR_VERSION), ('schema', SCHEMA_VERSION), ('schema source', 'benchmarks/schema/result-v1.schema.json')))}"
        + '</dl></section>'
        + '<section><h2>Suite budgets and selection</h2>'
        + '<table><caption>Manifest values</caption><thead><tr><th>Suite</th><th>Version</th><th>Budget seconds</th>'
        + '<th>Quality version</th><th>Performance cases</th><th>Engines</th></tr></thead>'
        + f"<tbody>{''.join(suite_rows)}</tbody></table></section>"
        + '<section><h2>Publication rules</h2><ul>'
        + '<li>Only schema-valid, complete, clean, quality-passing records with verified headline evidence are rendered.</li>'
        + '<li>Raw samples remain visible; summary statistics are median, minimum, maximum, MAD, and count.</li>'
        + '<li>Peer comparability and reasons come from <code>compare_records</code>; non-comparable pairs carry no relative value.</li>'
        + '<li>Every numeric metric links to run evidence containing commit, GPU, artifact, workload, correctness, clock, and cache identity.</li>'
        + '</ul></section></main>'
    )
    return page("Benchmark methodology", body, href_prefix="")


def _stylesheet() -> str:
    return """/* Deterministic benchmark site stylesheet; no runtime dependencies. */
:root { color-scheme: light dark; --bg: #10141d; --panel: #1b2230; --ink: #e7edf7; --muted: #a9b6c9; --accent: #82c7ff; --bad: #ff9c9c; --good: #9fe3b1; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 16px/1.5 system-ui, sans-serif; }
nav { display: flex; flex-wrap: wrap; gap: 1rem; padding: 1rem max(1rem, calc((100vw - 1100px) / 2)); background: #080b11; }
nav a, a { color: var(--accent); }
.container { max-width: 1100px; margin: 0 auto; padding: 2rem 1rem 4rem; }
section { margin: 2rem 0; padding: 1rem; background: var(--panel); border-radius: .4rem; }
h1, h2, h3 { line-height: 1.2; }
table { width: 100%; border-collapse: collapse; display: block; overflow-x: auto; }
caption { text-align: left; color: var(--muted); padding: .4rem 0; }
th, td { padding: .45rem .6rem; border-bottom: 1px solid #3a4658; text-align: left; vertical-align: top; }
th { white-space: nowrap; }
dl { display: grid; grid-template-columns: minmax(12rem, 22rem) 1fr; gap: .3rem 1rem; }
dt { color: var(--muted); }
dd { margin: 0; overflow-wrap: anywhere; }
.badges { display: flex; flex-wrap: wrap; gap: .5rem; }
.badge { display: inline-block; padding: .2rem .55rem; border-radius: 999px; background: #33415a; }
.badge.quality, .badge.comparable { background: #245a39; color: var(--good); }
.badge.qualified { background: #5a3b24; color: #ffd29a; }
.metric { font-size: 1.2rem; }
.metric-columns { display: grid; grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr)); gap: 1rem; }
pre { overflow-x: auto; padding: 1rem; background: #080b11; border-radius: .3rem; }
.empty { color: var(--muted); }
"""
