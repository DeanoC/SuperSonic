# FLM First-Class End-to-End Verifier Design

## Goal

Provide one reproducible command that proves the complete Qwen3.6 35B-A3B
native INT4 FLM path:

1. geo-quant exports a self-contained FLM from the Hugging Face source model;
2. geo-quant validates the file against the strict SuperSonic native INT4
   producer contract;
3. SuperSonic loads only that FLM and generates at least one token on the GPU;
4. the verifier checks structured evidence that the direct FLM path was used.

The verifier is a development and release contract between the producer and
consumer. It is not part of normal SuperSonic inference startup.

## Current State

SuperSonic `main` already accepts a Qwen3.6 35B-A3B FLM as `--model-dir`,
infers the model and weight mode from the file, loads native INT4 direct-plan
views, and records FLM evidence in the existing benchmark JSON. geo-quant
`main` already supports streaming Qwen3.6 INT4 FLM export, omitting Hugging
Face compatibility assets, and the
`supersonic-qwen36-moe-native-int4` validation profile.

The former canonical aligned artifact is no longer present on the workstation.
The surviving
`/mnt/data/runs/geo-quant/qwen36-35b-a3b-supersonic-native-int4.flm`
predates the finalized native-layout and direct-IO alignment contract. It fails
the current strict profile and must not be reused as evidence for the main
path.

## Ownership Boundary

The orchestration belongs in SuperSonic because it verifies a SuperSonic
acceptance contract. geo-quant remains an external producer invoked through its
public Python CLI. SuperSonic must not import geo-quant modules or copy its FLM
writer implementation.

The verifier accepts explicit paths for:

- the Hugging Face source snapshot;
- the geo-quant checkout and Python environment;
- the output FLM artifact;
- the SuperSonic executable;
- the benchmark JSON report.

Defaults may describe this workstation's development setup, but every path is
overridable. The verifier must not depend on an obsolete feature worktree.

## Artifact Lifecycle

The verifier uses an export-or-reuse policy:

1. If `--regenerate` is set, run the producer regardless of an existing file.
2. If the output path does not exist, run the producer.
3. If the output path exists, validate it with geo-quant's
   `supersonic-qwen36-moe-native-int4` profile.
4. Reuse the file only when strict validation succeeds.
5. If validation fails, preserve the existing file and report that it is stale
   or incompatible. A separate newly written output path is used for
   regeneration so a failed export cannot destroy the prior artifact.

The producer command streams from the Qwen3.6 35B-A3B Hugging Face snapshot,
writes only FLM output, omits optional Hugging Face JSON compatibility assets,
uses INT4 group size 128, and requests the strict SuperSonic validation profile
before returning success.

After export or reuse, the verifier runs strict validation again with BLAKE3
payload verification. This deliberately reads all payload bytes and is part of
the correctness gate, not the measured fast-load interval.

## Consumer Run

The verifier invokes the existing
`tests/gfx1100/bench_qwen36_he_supersonic.py` FLM target profile with:

- the FLM path as `--model-dir`;
- no explicit `--model` or quantization selection;
- one prompt and at least one generated token;
- stage timing and HAL profiling enabled;
- a caller-selected timeout and JSON output path.

The benchmark remains responsible for launching SuperSonic and collecting
stdout-derived fields. The verifier is responsible for deciding whether the
result proves the producer-to-consumer contract.

## Required Evidence

The verifier succeeds only when the benchmark process returns successfully and
the JSON report shows all of the following:

- model resolution came from the FLM runtime descriptor and equals
  `qwen3.6-35b-a3b`;
- every successful row reports `INT4 native FLM`;
- every successful row reaches FLM `ready-for-decode`;
- native INT4 direct-plan coverage is nonzero;
- BF16 fallback coverage is zero;
- at least one token was generated;
- the summary contains a nonzero FLM transfer byte count and measured transfer
  throughput;
- no benchmark validation errors are present.

The strict geo-quant profile is the producer-side structural contract. The
SuperSonic report is the consumer-side execution contract. Neither substitutes
for the other.

## Failure Handling

Failures must identify their phase: input discovery, producer export, strict
validation, SuperSonic execution, or report evidence. Subprocess failures
include the command and exit status. Timeouts name the phase and configured
timeout.

The verifier must never:

- silently use the Hugging Face snapshot during the SuperSonic run;
- silently select BF16 or a non-FLM weight path;
- accept the old artifact because it merely has an `.flm` suffix;
- overwrite a prior artifact before a replacement export and validation
  complete;
- fall back from an explicitly selected storage backend without reporting it.

Temporary export output is retained on failure for diagnosis. A validated
temporary output is atomically promoted to the requested artifact path.

## Structure

The Python verifier is split into focused functions:

- argument and environment-default parsing;
- producer export command construction;
- strict validation command construction;
- artifact decision logic;
- SuperSonic benchmark command construction;
- subprocess execution with phase-aware errors;
- benchmark report validation and concise summary output.

These functions keep policy testable without quantizing a model or requiring a
GPU in unit tests.

## Testing

Unit tests cover:

- command construction with fully explicit paths;
- missing artifact selects export;
- valid artifact selects reuse;
- `--regenerate` selects export;
- invalid artifact is preserved and selects safe regeneration;
- successful reports satisfy every required evidence field;
- missing model inference, native coverage, decode readiness, generated tokens,
  transfer evidence, or process success is rejected;
- producer, validator, benchmark, and timeout failures name the correct phase;
- the SuperSonic command contains no HF source path, explicit model, or INT4
  selection.

Focused existing benchmark and support-matrix tests remain part of verification.
The final manual gate runs on the workstation's ROCm GPU:

1. stream-export a fresh canonical FLM using merged geo-quant `main`;
2. verify all FLM payload hashes;
3. build the current SuperSonic release binary;
4. load only the FLM and generate at least one token;
5. retain the artifact and JSON report as the new reproducible baseline.

hipFile is outside this stage because the installed ROCm 7.1.1 stack lacks the
required AIS support. The verifier may forward an explicitly requested FLM
transfer backend, but its canonical success path uses the available pageable
H2D backend. Storage-direct performance becomes a separate follow-on once a
ROCm 7.2+ hipFile environment passes `ais-check`.

## Non-Goals

- Adding another model family or Qwen size.
- Changing the FLM binary format or producer ABI.
- Importing geo-quant as a SuperSonic library dependency.
- Making full payload verification part of normal fast startup.
- Claiming hipFile or storage-direct performance on the current ROCm stack.
- Measuring model quality beyond proving that real inference completes.
