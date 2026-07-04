# FLM First-Class Path Design

## Goal

Make an FLM file exported by geo-quant a first-class SuperSonic model source. A normal run should pass the FLM path as `--model-dir`, require no Hugging Face snapshot, require no quantization flag such as `--int4`, and load the executable weight mode directly from the file metadata.

## Original Gap

The current merged path can load the latest native INT4 Qwen3.6 MoE FLM only when the user also passes `--int4`. Without that flag SuperSonic opens the FLM, reads the runtime config and tokenizer, then rejects the file because the default CLI quantization profile is BF16. That makes the CLI flag, not the FLM, the source of truth.

The latest geo-quant native INT4 artifact proves the direct path exists: with `--int4`, SuperSonic reports `INT4 native FLM` and reaches the dry-run ready-for-decode path. The first-class gap is selection and ergonomics, not the underlying native tensor layout.

The e2e implementation also exposed a stricter direct-layout requirement: Qwen3.6 linear-attention raw tensors must be stored in the SuperSonic runtime layout inside the FLM. In particular, `linear_attn.conv1d.weight` is the squeezed depthwise-conv view, `linear_attn.dt_bias` is `[1, 1, H]`, and `linear_attn.A_log` is exponentiated BF16 `[1, 1, H]`. A runnable FLM should contain those direct values; SuperSonic labels and validates them, but does not reshape HF checkpoint payloads on the normal path.

## Implementation State

The active branch has closed the original source-selection gap for the
Qwen3.6 35B-A3B path: `--model-dir <file.flm>` is enough, `--model` can be
inferred from the FLM runtime descriptor, and no `--int4` flag is required.
The current measured path is still pageable host-to-device transfer from the
aligned FLM payload into the direct native INT4 runtime layout. Real
storage-to-device performance remains unproven on this workstation because the
host stack is Fedora 44 with ROCm/HIP 7.1.1 and no `hipfile.h`/`libhipfile`.

## Architecture

SuperSonic should treat `.flm` as authoritative for FLM-backed runs:

- `--model-dir /path/model.flm` or `--flm-file /path/model.flm` selects an FLM model source.
- FLM open options expose logical/direct INT4 aliases for FLM sources without needing a CLI quantization profile.
- Qwen3.6 MoE weight-mode selection probes the already-open FLM store and chooses native INT4 when the logical probe is `LayoutTag::Int4Quantized` with `u8`, or BF16 fallback when the probe is `LayoutTag::Raw` with `bf16`.
- Raw direct-plan aliases are labeled with their runtime layout tags when the FLM plan target shape already matches the runtime view, for example `DepthwiseConvSqueezed`, `HeadBiasReshaped`, and `HeadExpReshaped` for Qwen3.6 linear attention.
- Explicit incompatible quantization flags remain validation errors. The no-flag path is file-driven.

This keeps the direct path first-class while preserving fallbacks for FLM artifacts that carry CT-style or BF16 views.

## Transfer Strategy

The first-class FLM path should not treat file-backed `mmap` plus host
registration as the architectural endpoint. `mmap` remains the portable source
view for fast tensor-table access and normal host-to-device fallback. The
preferred SuperSonic path is still to load compatible FLM tensor extents into
the GPU-resident layout the execution plan consumes directly, such as Qwen3.6
MoE virtual expert slabs.

HIP host registration of mmap-backed payloads is useful as an opt-in diagnostic
or dense-fallback experiment, but it is not GPU-direct IO. Its cost depends on
filesystem cache state, page faults, driver page pinning, and concurrent GPU
memory pressure. Therefore the normal FLM path must not enable registered mmap
uploads by default.

The next fast-load boundary should be a loader transfer abstraction that can use
the same FLM tensor extent metadata with multiple implementations:

- pageable host-to-device copy as the portable fallback;
- explicitly pinned or registered host staging for experiments and fallback
  paths;
- ROCm `hipFile` storage-to-device transfer when ROCm 7.2+ hipFile is
  installed and the file/device/filesystem constraints are satisfied.

On ROCm, the storage-direct backend is specifically `hipFile`, AMD's cuFile-like
API. SuperSonic should detect `hipfile.h` and `libhipfile` at build time and
compile a small bridge that reads FLM file ranges directly into device memory.
When hipFile is absent, as on the current Fedora 44 ROCm 7.1.1 setup, the
`GpuDirectStorage` backend must return `Unsupported` before mapping VMM pages or
falling back to pageable H2D. A real fast-path verification requires ROCm 7.2+
hipFile, `ais-check` passing, and the FLM payload on a supported local NVMe
filesystem such as ext4 or xfs.

The hipFile bridge must run in strict direct mode when selected by SuperSonic:
it disables hipFile compatibility fallback with `hipFileSetParameterBool` before
opening the driver path, opens the FLM with `O_DIRECT`, registers the target
device buffer range with `hipFileBufRegister`, performs the `hipFileRead`, and
deregisters the buffer before returning. This keeps the explicit
`GpuDirectStorage` backend from being satisfied by hipFile's POSIX fallback or
an internal bounce path while we are trying to measure direct storage-to-device
load performance.

The runtime selector is `--flm-virtual-transfer-backend`, with
`SUPERSONIC_FLM_VIRTUAL_TRANSFER_BACKEND` kept as an environment fallback. It
defaults to `pageable-h2d`; `gpu-direct-storage`, `gds`, or `hipfile` selects
the named storage-to-device backend for MoE expert virtual arena loads, covering
both sparse residency page-in and eager virtual expert slab loads. This selector
is deliberately explicit while hipFile availability is platform-dependent, and
the CLI value wins over the env var when both are present. If it is forced on a
build without hipFile support, the loader must fail before pageable mapping/copy
rather than timing an accidental fallback. Selecting `hipfile` without
`SUPERSONIC_MOE_ISLAND_CAP_EXPERTS` forces the eager routed expert VMM path so a
dense-fit model cannot silently bypass the requested storage-to-device backend.
Current sparse async MoE page-in uses pinned host staging, so the `hipfile`
backend disables that async staging path until direct asynchronous storage reads
are implemented.

SuperSonic should expose a concrete storage-extent descriptor for direct
file-backed tensors before adding transfer backends. The descriptor names the
source file, byte offset, byte length, storage dtype/shape/layout, and runtime
upload dtype/shape. Synthesized or transformed fallback aliases must not expose
a single direct extent. This keeps future GPU-direct work from inferring file
identity through mmap pointers or conflating packed storage shape with the
execution view.

For Qwen3.6 MoE, the highest-value transfer boundary is the virtual arena range
loader, not only dense `GpuBuffer` construction: routed expert slabs already
enter execution through stable virtual allocations. A GPU-direct backend should
therefore target extent-to-virtual-allocation range loads so the direct FLM plan
can become resident in the execution layout without a dense staging detour.

The `qwen36_flm_upload_probe` binary also has focused storage-direct modes for
ROCm 7.2 bring-up. `--only-storage-direct` asks `BakedStore` for the selected
tensor's absolute FLM file range and calls the same HAL
`copy_storage_to_device` boundary used by the virtual arena backend, without
measuring pageable or pinned baselines first. On builds without hipFile support
this mode must fail with the explicit hipFile requirement; on a ROCm 7.2+
hipFile system it should emit `copy_storage_to_device_*` profile fields. Use the
older `--storage-direct` mode when the same run should compare pageable and
pinned H2D baselines before the storage-direct attempt. Because strict hipFile
uses `O_DIRECT`, the probe rejects storage-direct ranges whose file offset or
length is not 4 KiB aligned. Small tensors such as `linear_attn.dt_bias` remain
valid FLM payloads, but they are not standalone tensor-granular hipFile transfer
candidates; use block-aligned expert slabs for storage-direct bring-up.

FLM itself should describe the tensor storage extents, layout ABI, direct plan,
and integrity information. The inference engine chooses the best transfer
backend for the current platform without changing FLM semantics.

## Data Flow

1. CLI resolves an effective FLM source from `--flm-file` or `.flm` `--model-dir`.
2. SuperSonic opens the FLM once with runtime aliases enabled and optional BLAKE3 payload verification based on `--verify-flm-hashes`.
3. Runtime config and tokenizer are loaded from FLM assets.
4. Qwen3.6 MoE loader probes the logical weight view in the open store and selects the runtime weight mode from the file.
5. Decode uses the selected mode and the already-open FLM store. No HF `config.json`, `tokenizer.json`, safetensors, fetch, or bake path is consulted.

## Error Handling

If a Qwen3.6 MoE FLM has no recognized executable probe, the loader should fail with a message naming the missing/incompatible probe. If the user asks for an incompatible non-INT4 quant mode with an FLM source, the CLI should continue to reject it before load. Payload hash verification remains opt-in because it reads the payload bytes and is not representative of fast-load latency.

## Testing

The primary red test is the current failing behavior: the latest geo-quant native INT4 FLM must dry-run under SuperSonic without `--int4`.

Automated coverage should include:

- unit tests proving FLM open options enable runtime aliases independent of CLI quant flags;
- unit tests proving Qwen3.6 MoE FLM weight selection is file-probe driven;
- model-store tests proving Qwen3.6 linear-attention direct raw values match the SuperSonic runtime bake byte-for-byte;
- runner tests proving `--flm-virtual-transfer-backend` / `SUPERSONIC_FLM_VIRTUAL_TRANSFER_BACKEND` parse direct-storage aliases, CLI selection overrides the env fallback, and forced `GpuDirectStorage` does not fall back to pageable mapping/copy when unsupported;
- upload-probe tests proving `--storage-direct` is explicit and records `copy_storage_to_device` timing/byte counters separately from H2D counters;
- env-gated integration tests that run the real 35B A3B FLM with no HF snapshot and no `--int4`;
- docs with canonical geo-quant export and SuperSonic run commands.

Manual/e2e verification should use the current 35B native artifact and capture `--emit-stage-timings` output so load and inference speeds are measurable.
