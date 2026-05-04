use std::borrow::Cow;

use anyhow::{anyhow, Context, Result};
use gpu_hal::{GpuBuffer, ScalarType, VirtualAllocationRole, VirtualArena};
use model_store::manifest::LayoutTag;
use model_store::BakedStore;
use qwen36_moe::config::TextConfig;

use crate::qwen36_moe_residency::{MoeExpertProjection, MoeExpertResidencyManager};
use crate::qwen36_moe_types::{
    AttnLayerBuffers, FfnInt4Sidecars, FfnLayerBuffers, FullAttnInt4Sidecars, FullAttnKvCache,
    LayerBuffers, LinearAttnInt4Sidecars, MultiLayerGeom, ResidentWeight,
};

/// Open a BakedStore from the bake dir, loading one tensor by name to a
/// fresh GpuBuffer. The wrapper exists to attach a useful context message
/// when a tensor is missing (the bake-validation in `inspect_bake` already
/// runs as part of the dry-run, so a missing tensor here is a real bug).
pub(crate) fn load_to_gpu(store: &BakedStore, ordinal: usize, name: &str) -> Result<GpuBuffer> {
    let resolved = resolve_qwen36_store_name(store, name);
    store
        .load_to_gpu(resolved.as_ref(), ordinal)
        .with_context(|| format!("BakedStore::load_to_gpu({name})"))
}

fn load_to_resident_weight(
    store: &BakedStore,
    ordinal: usize,
    name: &str,
) -> Result<ResidentWeight> {
    load_to_gpu(store, ordinal, name).map(ResidentWeight::Dense)
}

fn load_to_virtual_resident_weight(
    store: &BakedStore,
    arena: &mut VirtualArena,
    name: &str,
) -> Result<ResidentWeight> {
    let resolved = resolve_qwen36_store_name(store, name);
    let id = store
        .load_to_virtual_arena(arena, resolved.as_ref(), VirtualAllocationRole::MoeExpert)
        .with_context(|| format!("BakedStore::load_to_virtual_arena({name})"))?;
    let allocation = arena
        .allocation(id)
        .ok_or_else(|| anyhow!("virtual allocation id {id} disappeared after loading {name}"))?;
    let buffer = allocation.buffer();
    Ok(ResidentWeight::Virtual {
        allocation_id: id,
        ptr: buffer.as_ptr(),
        dtype: buffer.dtype(),
        shape: buffer.shape().to_vec(),
        len_bytes: buffer.len_bytes(),
    })
}

fn load_to_sparse_resident_weight(
    store: &BakedStore,
    manager: &mut MoeExpertResidencyManager,
    layer_idx: usize,
    projection: MoeExpertProjection,
    name: &str,
    expert_count: usize,
) -> Result<ResidentWeight> {
    let resolved = resolve_qwen36_store_name(store, name);
    manager
        .register_tensor(
            store,
            layer_idx,
            projection,
            resolved.as_ref(),
            expert_count,
        )
        .with_context(|| format!("reserve sparse MoE expert tensor {name}"))?;
    manager
        .resident_weight(layer_idx, projection)
        .with_context(|| format!("build sparse resident weight for {name}"))
}

pub(crate) fn store_contains_qwen36(store: &BakedStore, name: &str) -> bool {
    store.contains(resolve_qwen36_store_name(store, name).as_ref())
}

pub(crate) fn store_layout_qwen36<'a>(store: &'a BakedStore, name: &str) -> Option<&'a LayoutTag> {
    store.layout(resolve_qwen36_store_name(store, name).as_ref())
}

pub(crate) fn resolve_qwen36_store_name<'a>(store: &BakedStore, name: &'a str) -> Cow<'a, str> {
    if store.contains(name) {
        return Cow::Borrowed(name);
    }
    if name.contains(".mlp.experts.") {
        if let Some(rest) = name.strip_prefix("model.language_model.") {
            let alt = format!("model.{rest}");
            if store.contains(&alt) {
                return Cow::Owned(alt);
            }
        }
    }
    Cow::Borrowed(name)
}

/// Pinned by `oracle/bake_int4.py` and the kernel — every INT4 tensor in
/// the bake quantizes at this group size. Detected per-tensor via a
/// `*_int4_scale` sidecar; if any quantizable tensor is present and
/// uses a different group_size we'd surface that as an error.
pub(crate) const QWEN36_MOE_INT4_GROUP_SIZE: i32 = 128;
const QWEN36_MOE_FP8_BLOCK_SIZE: i32 = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Qwen36WeightMode {
    Bf16,
    Int4,
    Fp8,
}

impl Qwen36WeightMode {
    pub(crate) fn is_int4(self) -> bool {
        self == Self::Int4
    }

    pub(crate) fn display_name(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::Int4 => "INT4 GPTQ",
            Self::Fp8 => "FP8 native",
        }
    }
}

/// Build one layer's worth of GPU-resident weight + state buffers from a
/// BakedStore. Decides full-attn vs linear-attn by consulting the config's
/// `layer_types` (every 4th layer is full per the standard hybrid pattern).
/// `weight_mode` controls whether the weight tensors are loaded as BF16,
/// packed INT4 nibbles + sidecars, or native FP8 bytes + scale sidecars. The
/// INT4 bake naming convention pairs
/// `<name>.weight` (packed) with `<name>.weight_int4_scale`/`_int4_zero`
/// for dense projections, and `<name>` (packed) with `<name>_int4_scale`/
/// `_int4_zero` for fused-expert tensors (no `.weight` suffix in the
/// HuggingFace checkpoint). FP8 uses the same weight names plus
/// `<name>_scale_inv`.
pub(crate) fn load_layer_buffers(
    store: &BakedStore,
    ordinal: usize,
    layer_idx: usize,
    geom: &MultiLayerGeom,
    text_config: &TextConfig,
    weight_prefix: &str,
    weight_mode: Qwen36WeightMode,
    // When > 0, allocate a KV cache for full-attention layers sized for
    // `kv_max_t` past tokens. Linear-attn layers use `conv_state` +
    // `recurrent_state` instead (always allocated). 0 = no KV cache,
    // kernel falls back to kv_len=1 (back-compat for the parity test).
    kv_max_t: usize,
    // When true, the layer's KV cache is allocated as FP8 E4M3 bytes
    // with F32 per-(head, position) scales and an optional BF16
    // sidecar (gated by `qwen35::state::kv_fp8_bf16_sidecar_*` env
    // helpers).
    kv_fp8: bool,
    // When true, full-attention K/V caches use VMM reservations for
    // their main K/V storage. Applies to BF16 and FP8 KV; FP8 scale
    // and sidecar buffers stay dense.
    kv_vmm: bool,
    mut expert_arena: Option<&mut VirtualArena>,
    mut expert_residency: Option<&mut MoeExpertResidencyManager>,
) -> Result<LayerBuffers> {
    let lp = format!("{weight_prefix}.layers.{layer_idx}");

    let attn = if text_config.is_full_attention(layer_idx) {
        let fa = format!("{lp}.self_attn");
        let int4 = match weight_mode {
            Qwen36WeightMode::Int4 => Some(FullAttnInt4Sidecars {
                group_size: QWEN36_MOE_INT4_GROUP_SIZE,
                q_proj_scale: load_to_gpu(
                    store,
                    ordinal,
                    &format!("{fa}.q_proj.weight_int4_scale"),
                )?,
                q_proj_zero: load_to_gpu(store, ordinal, &format!("{fa}.q_proj.weight_int4_zero"))?,
                k_proj_scale: load_to_gpu(
                    store,
                    ordinal,
                    &format!("{fa}.k_proj.weight_int4_scale"),
                )?,
                k_proj_zero: load_to_gpu(store, ordinal, &format!("{fa}.k_proj.weight_int4_zero"))?,
                v_proj_scale: load_to_gpu(
                    store,
                    ordinal,
                    &format!("{fa}.v_proj.weight_int4_scale"),
                )?,
                v_proj_zero: load_to_gpu(store, ordinal, &format!("{fa}.v_proj.weight_int4_zero"))?,
                o_proj_scale: load_to_gpu(
                    store,
                    ordinal,
                    &format!("{fa}.o_proj.weight_int4_scale"),
                )?,
                o_proj_zero: load_to_gpu(store, ordinal, &format!("{fa}.o_proj.weight_int4_zero"))?,
            }),
            Qwen36WeightMode::Fp8 => Some(FullAttnInt4Sidecars {
                group_size: -QWEN36_MOE_FP8_BLOCK_SIZE,
                q_proj_scale: load_to_gpu(
                    store,
                    ordinal,
                    &format!("{fa}.q_proj.weight_scale_inv"),
                )?,
                q_proj_zero: GpuBuffer::zeros(ordinal, ScalarType::U8, &[1])?,
                k_proj_scale: load_to_gpu(
                    store,
                    ordinal,
                    &format!("{fa}.k_proj.weight_scale_inv"),
                )?,
                k_proj_zero: GpuBuffer::zeros(ordinal, ScalarType::U8, &[1])?,
                v_proj_scale: load_to_gpu(
                    store,
                    ordinal,
                    &format!("{fa}.v_proj.weight_scale_inv"),
                )?,
                v_proj_zero: GpuBuffer::zeros(ordinal, ScalarType::U8, &[1])?,
                o_proj_scale: load_to_gpu(
                    store,
                    ordinal,
                    &format!("{fa}.o_proj.weight_scale_inv"),
                )?,
                o_proj_zero: GpuBuffer::zeros(ordinal, ScalarType::U8, &[1])?,
            }),
            Qwen36WeightMode::Bf16 => None,
        };
        // KV cache: allocate per-layer when multi-token decode is requested.
        // BF16 path: [kv_max_t, num_kv_heads * head_dim] BF16 for K and V.
        // FP8 path: same shape but U8 (FP8 E4M3 bytes) plus F32 scales
        // [num_kv_heads, kv_max_t] for K and V, plus optional BF16 sidecar
        // [num_kv_heads, sidecar_window, head_dim] when enabled.
        // VMM path (default on HIP when supported, or
        // SUPERSONIC_VMM_KV=1): K/V are VirtualBuffer reservations with a
        // mapped prefix; dense k/v are None. FP8 scale and sidecar buffers
        // remain regular dense GpuBuffers.
        let kv_dim = (geom.num_kv_heads as usize) * (geom.head_dim as usize);
        let kv_cache = if kv_max_t > 0 {
            let num_kv_heads = geom.num_kv_heads as usize;
            let head_dim = geom.head_dim as usize;

            let (
                k,
                v,
                virtual_kv_cache_k,
                virtual_kv_cache_v,
                virtual_kv_max_t,
                kv_scale_k,
                kv_scale_v,
            ) = if kv_vmm {
                // VMM-backed: reserve full logical capacity AND map all of
                // it up front. We don't evict Qwen3.6 KV in v1 (CpuBackup is
                // off — VirtualBacking::Discard), and the kernel writes K/V
                // at arbitrary `eff_cache_pos` between 0 and kv_max_t-1, so
                // any unmapped page would page-fault mid-launch. The benefit
                // here is stable VA and shared VMM descriptor plumbing, not
                // decode-time KV paging yet.
                let kv_dtype = if kv_fp8 {
                    ScalarType::U8
                } else {
                    ScalarType::BF16
                };
                let kv_dtype_bytes = if kv_fp8 { 1 } else { 2 };
                let map_full_bytes = kv_max_t * kv_dim * kv_dtype_bytes;
                let vk = gpu_hal::VirtualBuffer::reserve_and_map_prefix(
                    ordinal,
                    kv_dtype,
                    &[kv_max_t, kv_dim],
                    map_full_bytes,
                    gpu_hal::VirtualBacking::Discard,
                )
                .with_context(|| format!("vmm reserve kv_cache_k (layer {layer_idx})"))?;
                let vv = gpu_hal::VirtualBuffer::reserve_and_map_prefix(
                    ordinal,
                    kv_dtype,
                    &[kv_max_t, kv_dim],
                    map_full_bytes,
                    gpu_hal::VirtualBacking::Discard,
                )
                .with_context(|| format!("vmm reserve kv_cache_v (layer {layer_idx})"))?;
                let (sk, sv) = if kv_fp8 {
                    let sk = GpuBuffer::zeros(ordinal, ScalarType::F32, &[num_kv_heads, kv_max_t])
                        .with_context(|| format!("alloc kv_scale_k (layer {layer_idx})"))?;
                    let sv = GpuBuffer::zeros(ordinal, ScalarType::F32, &[num_kv_heads, kv_max_t])
                        .with_context(|| format!("alloc kv_scale_v (layer {layer_idx})"))?;
                    (Some(sk), Some(sv))
                } else {
                    (None, None)
                };
                (None, None, Some(vk), Some(vv), Some(kv_max_t), sk, sv)
            } else if kv_fp8 {
                // dense FP8 path
                let k = GpuBuffer::zeros(ordinal, ScalarType::U8, &[kv_max_t, kv_dim])
                    .with_context(|| format!("alloc kv_cache_k FP8 (layer {layer_idx})"))?;
                let v = GpuBuffer::zeros(ordinal, ScalarType::U8, &[kv_max_t, kv_dim])
                    .with_context(|| format!("alloc kv_cache_v FP8 (layer {layer_idx})"))?;
                let sk = GpuBuffer::zeros(ordinal, ScalarType::F32, &[num_kv_heads, kv_max_t])
                    .with_context(|| format!("alloc kv_scale_k (layer {layer_idx})"))?;
                let sv = GpuBuffer::zeros(ordinal, ScalarType::F32, &[num_kv_heads, kv_max_t])
                    .with_context(|| format!("alloc kv_scale_v (layer {layer_idx})"))?;
                (Some(k), Some(v), None, None, None, Some(sk), Some(sv))
            } else {
                // BF16 path
                let k = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[kv_max_t, kv_dim])
                    .with_context(|| format!("alloc kv_cache_k (layer {layer_idx})"))?;
                let v = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[kv_max_t, kv_dim])
                    .with_context(|| format!("alloc kv_cache_v (layer {layer_idx})"))?;
                (Some(k), Some(v), None, None, None, None, None)
            };
            let (kv_shadow_k, kv_shadow_v, kv_shadow_window) =
                if kv_fp8 && qwen35::state::kv_fp8_bf16_sidecar_enabled() {
                    let window = qwen35::state::kv_fp8_bf16_sidecar_window_tokens()
                        .unwrap_or(kv_max_t)
                        .min(kv_max_t);
                    if window == 0 {
                        (None, None, 0)
                    } else {
                        let sk = GpuBuffer::zeros(
                            ordinal,
                            ScalarType::BF16,
                            &[num_kv_heads, window, head_dim],
                        )
                        .with_context(|| format!("alloc kv_shadow_k (layer {layer_idx})"))?;
                        let sv = GpuBuffer::zeros(
                            ordinal,
                            ScalarType::BF16,
                            &[num_kv_heads, window, head_dim],
                        )
                        .with_context(|| format!("alloc kv_shadow_v (layer {layer_idx})"))?;
                        (Some(sk), Some(sv), window as i32)
                    }
                } else {
                    (None, None, 0)
                };
            Some(FullAttnKvCache {
                k,
                v,
                kv_max_t: kv_max_t as i32,
                kv_scale_k,
                kv_scale_v,
                kv_shadow_k,
                kv_shadow_v,
                kv_shadow_start: -1,
                kv_shadow_window,
                virtual_kv_cache_k,
                virtual_kv_cache_v,
                virtual_kv_max_t,
            })
        } else {
            None
        };
        AttnLayerBuffers::Full {
            input_norm_w: load_to_gpu(store, ordinal, &format!("{lp}.input_layernorm.weight"))?,
            q_proj_w: load_to_gpu(store, ordinal, &format!("{fa}.q_proj.weight"))?,
            k_proj_w: load_to_gpu(store, ordinal, &format!("{fa}.k_proj.weight"))?,
            v_proj_w: load_to_gpu(store, ordinal, &format!("{fa}.v_proj.weight"))?,
            q_norm_w: load_to_gpu(store, ordinal, &format!("{fa}.q_norm.weight"))?,
            k_norm_w: load_to_gpu(store, ordinal, &format!("{fa}.k_norm.weight"))?,
            o_proj_w: load_to_gpu(store, ordinal, &format!("{fa}.o_proj.weight"))?,
            int4,
            kv_cache,
        }
    } else {
        let la = format!("{lp}.linear_attn");
        let kernel = geom.conv_kernel_dim as usize;
        let key_dim = (geom.num_k_heads as usize) * (geom.head_k_dim as usize);
        let val_dim = (geom.num_v_heads as usize) * (geom.head_v_dim as usize);
        let qkv_dim = 2 * key_dim + val_dim;
        let state_elems =
            (geom.num_v_heads as usize) * (geom.head_k_dim as usize) * (geom.head_v_dim as usize);

        // First-decode-token state: conv + recurrent both zeros. The kernel
        // mutates them in place; PR 4d will persist them across decode steps.
        let conv_state = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[qkv_dim, kernel - 1])
            .with_context(|| format!("alloc conv_state (layer {layer_idx})"))?;
        let recurrent_state = GpuBuffer::zeros(ordinal, ScalarType::F32, &[state_elems])
            .with_context(|| format!("alloc recurrent_state (layer {layer_idx})"))?;

        let int4 = match weight_mode {
            Qwen36WeightMode::Int4 => Some(LinearAttnInt4Sidecars {
                group_size: QWEN36_MOE_INT4_GROUP_SIZE,
                in_proj_qkv_scale: load_to_gpu(
                    store,
                    ordinal,
                    &format!("{la}.in_proj_qkv.weight_int4_scale"),
                )?,
                in_proj_qkv_zero: load_to_gpu(
                    store,
                    ordinal,
                    &format!("{la}.in_proj_qkv.weight_int4_zero"),
                )?,
                in_proj_z_scale: load_to_gpu(
                    store,
                    ordinal,
                    &format!("{la}.in_proj_z.weight_int4_scale"),
                )?,
                in_proj_z_zero: load_to_gpu(
                    store,
                    ordinal,
                    &format!("{la}.in_proj_z.weight_int4_zero"),
                )?,
                out_proj_scale: load_to_gpu(
                    store,
                    ordinal,
                    &format!("{la}.out_proj.weight_int4_scale"),
                )?,
                out_proj_zero: load_to_gpu(
                    store,
                    ordinal,
                    &format!("{la}.out_proj.weight_int4_zero"),
                )?,
            }),
            Qwen36WeightMode::Fp8 => Some(LinearAttnInt4Sidecars {
                group_size: -QWEN36_MOE_FP8_BLOCK_SIZE,
                in_proj_qkv_scale: load_to_gpu(
                    store,
                    ordinal,
                    &format!("{la}.in_proj_qkv.weight_scale_inv"),
                )?,
                in_proj_qkv_zero: GpuBuffer::zeros(ordinal, ScalarType::U8, &[1])?,
                in_proj_z_scale: load_to_gpu(
                    store,
                    ordinal,
                    &format!("{la}.in_proj_z.weight_scale_inv"),
                )?,
                in_proj_z_zero: GpuBuffer::zeros(ordinal, ScalarType::U8, &[1])?,
                out_proj_scale: load_to_gpu(
                    store,
                    ordinal,
                    &format!("{la}.out_proj.weight_scale_inv"),
                )?,
                out_proj_zero: GpuBuffer::zeros(ordinal, ScalarType::U8, &[1])?,
            }),
            Qwen36WeightMode::Bf16 => None,
        };

        AttnLayerBuffers::Linear {
            input_norm_w: load_to_gpu(store, ordinal, &format!("{lp}.input_layernorm.weight"))?,
            in_proj_qkv_w: load_to_gpu(store, ordinal, &format!("{la}.in_proj_qkv.weight"))?,
            in_proj_z_w: load_to_gpu(store, ordinal, &format!("{la}.in_proj_z.weight"))?,
            in_proj_a_w: load_to_gpu(store, ordinal, &format!("{la}.in_proj_a.weight"))?,
            in_proj_b_w: load_to_gpu(store, ordinal, &format!("{la}.in_proj_b.weight"))?,
            conv1d_w: load_to_gpu(store, ordinal, &format!("{la}.conv1d.weight"))?,
            // conv1d.bias may be absent — match the loader's behaviour.
            conv1d_bias: store
                .contains(&format!("{la}.conv1d.bias"))
                .then(|| load_to_gpu(store, ordinal, &format!("{la}.conv1d.bias")))
                .transpose()?,
            dt_bias: load_to_gpu(store, ordinal, &format!("{la}.dt_bias"))?,
            a_log: load_to_gpu(store, ordinal, &format!("{la}.A_log"))?,
            norm_w: load_to_gpu(store, ordinal, &format!("{la}.norm.weight"))?,
            out_proj_w: load_to_gpu(store, ordinal, &format!("{la}.out_proj.weight"))?,
            conv_state,
            recurrent_state,
            int4,
        }
    };

    let mp = format!("{lp}.mlp");
    // Fused-expert sidecars use `_int4_scale`/`_int4_zero` (no `.weight`).
    // Shared-expert MLPs use the dense `<name>.weight_int4_scale` form.
    let ffn_int4 = match weight_mode {
        Qwen36WeightMode::Int4 => Some(FfnInt4Sidecars {
            group_size: QWEN36_MOE_INT4_GROUP_SIZE,
            gate_up_proj_scale: load_to_gpu(
                store,
                ordinal,
                &format!("{mp}.experts.gate_up_proj_int4_scale"),
            )?,
            gate_up_proj_zero: load_to_gpu(
                store,
                ordinal,
                &format!("{mp}.experts.gate_up_proj_int4_zero"),
            )?,
            down_proj_scale: load_to_gpu(
                store,
                ordinal,
                &format!("{mp}.experts.down_proj_int4_scale"),
            )?,
            down_proj_zero: load_to_gpu(
                store,
                ordinal,
                &format!("{mp}.experts.down_proj_int4_zero"),
            )?,
            shared_gate_proj_scale: load_to_gpu(
                store,
                ordinal,
                &format!("{mp}.shared_expert.gate_proj.weight_int4_scale"),
            )?,
            shared_gate_proj_zero: load_to_gpu(
                store,
                ordinal,
                &format!("{mp}.shared_expert.gate_proj.weight_int4_zero"),
            )?,
            shared_up_proj_scale: load_to_gpu(
                store,
                ordinal,
                &format!("{mp}.shared_expert.up_proj.weight_int4_scale"),
            )?,
            shared_up_proj_zero: load_to_gpu(
                store,
                ordinal,
                &format!("{mp}.shared_expert.up_proj.weight_int4_zero"),
            )?,
            shared_down_proj_scale: load_to_gpu(
                store,
                ordinal,
                &format!("{mp}.shared_expert.down_proj.weight_int4_scale"),
            )?,
            shared_down_proj_zero: load_to_gpu(
                store,
                ordinal,
                &format!("{mp}.shared_expert.down_proj.weight_int4_zero"),
            )?,
        }),
        Qwen36WeightMode::Fp8 => Some(FfnInt4Sidecars {
            group_size: -QWEN36_MOE_FP8_BLOCK_SIZE,
            gate_up_proj_scale: load_to_gpu(
                store,
                ordinal,
                &format!("{mp}.experts.gate_up_proj_scale_inv"),
            )?,
            gate_up_proj_zero: GpuBuffer::zeros(ordinal, ScalarType::U8, &[1])?,
            down_proj_scale: load_to_gpu(
                store,
                ordinal,
                &format!("{mp}.experts.down_proj_scale_inv"),
            )?,
            down_proj_zero: GpuBuffer::zeros(ordinal, ScalarType::U8, &[1])?,
            shared_gate_proj_scale: load_to_gpu(
                store,
                ordinal,
                &format!("{mp}.shared_expert.gate_proj.weight_scale_inv"),
            )?,
            shared_gate_proj_zero: GpuBuffer::zeros(ordinal, ScalarType::U8, &[1])?,
            shared_up_proj_scale: load_to_gpu(
                store,
                ordinal,
                &format!("{mp}.shared_expert.up_proj.weight_scale_inv"),
            )?,
            shared_up_proj_zero: GpuBuffer::zeros(ordinal, ScalarType::U8, &[1])?,
            shared_down_proj_scale: load_to_gpu(
                store,
                ordinal,
                &format!("{mp}.shared_expert.down_proj.weight_scale_inv"),
            )?,
            shared_down_proj_zero: GpuBuffer::zeros(ordinal, ScalarType::U8, &[1])?,
        }),
        Qwen36WeightMode::Bf16 => None,
    };
    let ffn = FfnLayerBuffers {
        post_attn_norm_w: load_to_gpu(
            store,
            ordinal,
            &format!("{lp}.post_attention_layernorm.weight"),
        )?,
        gate_w: load_to_gpu(store, ordinal, &format!("{mp}.gate.weight"))?,
        // Note: experts.gate_up_proj / experts.down_proj have NO `.weight`
        // suffix in the published checkpoint — see expected_tensor_specs.
        gate_up_proj_w: if let Some(manager) = expert_residency.as_mut() {
            load_to_sparse_resident_weight(
                store,
                &mut **manager,
                layer_idx,
                MoeExpertProjection::GateUp,
                &format!("{mp}.experts.gate_up_proj"),
                geom.num_experts as usize,
            )?
        } else {
            match expert_arena.as_mut() {
                Some(arena) => load_to_virtual_resident_weight(
                    store,
                    &mut **arena,
                    &format!("{mp}.experts.gate_up_proj"),
                )?,
                None => {
                    load_to_resident_weight(store, ordinal, &format!("{mp}.experts.gate_up_proj"))?
                }
            }
        },
        down_proj_w: if let Some(manager) = expert_residency.as_mut() {
            load_to_sparse_resident_weight(
                store,
                &mut **manager,
                layer_idx,
                MoeExpertProjection::Down,
                &format!("{mp}.experts.down_proj"),
                geom.num_experts as usize,
            )?
        } else {
            match expert_arena.as_mut() {
                Some(arena) => load_to_virtual_resident_weight(
                    store,
                    &mut **arena,
                    &format!("{mp}.experts.down_proj"),
                )?,
                None => {
                    load_to_resident_weight(store, ordinal, &format!("{mp}.experts.down_proj"))?
                }
            }
        },
        shared_gate_proj_w: load_to_gpu(
            store,
            ordinal,
            &format!("{mp}.shared_expert.gate_proj.weight"),
        )?,
        shared_up_proj_w: load_to_gpu(
            store,
            ordinal,
            &format!("{mp}.shared_expert.up_proj.weight"),
        )?,
        shared_down_proj_w: load_to_gpu(
            store,
            ordinal,
            &format!("{mp}.shared_expert.down_proj.weight"),
        )?,
        shared_expert_gate_w: load_to_gpu(
            store,
            ordinal,
            &format!("{mp}.shared_expert_gate.weight"),
        )?,
        int4: ffn_int4,
    };

    Ok(LayerBuffers { attn, ffn })
}

pub(crate) fn load_all_layer_buffers(
    store: &BakedStore,
    ordinal: usize,
    geom: &MultiLayerGeom,
    text_config: &TextConfig,
    weight_prefix: &str,
    weight_mode: Qwen36WeightMode,
    kv_max_t: usize,
    kv_fp8: bool,
    kv_vmm: bool,
    mut expert_arena: Option<&mut VirtualArena>,
    mut expert_residency: Option<&mut MoeExpertResidencyManager>,
) -> Result<Vec<LayerBuffers>> {
    let mut layers = Vec::with_capacity(geom.num_layers as usize);
    for li in 0..geom.num_layers as usize {
        let layer = load_layer_buffers(
            store,
            ordinal,
            li,
            geom,
            text_config,
            weight_prefix,
            weight_mode,
            kv_max_t,
            kv_fp8,
            kv_vmm,
            expert_arena.as_deref_mut(),
            expert_residency.as_deref_mut(),
        )
        .with_context(|| format!("load layer {li} weights"))?;
        layers.push(layer);
    }
    Ok(layers)
}
