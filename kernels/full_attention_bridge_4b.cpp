#include "full_attention_4b.hip"

// This bridge serves the Qwen3.8 product. qwen35-prefixed kernel/C symbols
// remain only as historical bridge ABI spellings and are not compatibility
// aliases for another model.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <mutex>
#include <stdint.h>
#include <vector>

extern "C" int supersonic_prefill_encode_bridge_status(
    int project_status,
    int native_status);

extern "C" int supersonic_qwen35_4b_bf16_matmul_bridge_status(
    int project_status,
    int native_status) {
    return supersonic_prefill_encode_bridge_status(project_status, native_status);
}

extern "C" void supersonic_gqh_hip_lock();
extern "C" void supersonic_gqh_hip_unlock();
extern "C" int supersonic_gqh_hip_restore_planar(
    int device_ordinal,
    int rung,
    void* wire,
    int in_dim,
    int out_dim);
extern "C" int supersonic_qwen35_4b_hip_invalidate_decode_cache(
    int device_ordinal,
    const void* layers,
    const void* int4);
extern "C" [[noreturn]] void supersonic_gpu_integrity_fail_stop(
    const char* operation,
    int status,
    int device_ordinal);

namespace {

// Per-model launch preset, set once at startup by the Rust registry via
// `supersonic_qwen35_4b_hip_set_launch_preset`. Read by the persistent-decode
// bridge when the user hasn't supplied `SUPERSONIC_QWEN4B_BLOCKS`. Zero
// means "no preset, use the hardcoded gfx11xx default".
int g_preset_blocks = 0;
int g_preset_coop = 0;
bool g_hip_gqh_prepare_only = false;

inline void qwen4b_get_launch_preset(int& blocks, int& coop) {
    blocks = g_preset_blocks;
    coop = g_preset_coop;
}

} // anonymous namespace

extern "C" void supersonic_qwen35_4b_hip_set_launch_preset(int blocks, int coop) {
    supersonic_gqh_hip_lock();
    g_preset_blocks = blocks;
    g_preset_coop = coop;
    supersonic_gqh_hip_unlock();
}

extern "C" void supersonic_qwen35_4b_hip_set_gqh_prepare_only(int on) {
    supersonic_gqh_hip_lock();
    g_hip_gqh_prepare_only = on != 0;
    supersonic_gqh_hip_unlock();
}

extern "C" int supersonic_qwen35_4b_hip_matmul_int4_dequant(
    int dtype,
    size_t device_ordinal,
    size_t batch_elems,
    int m,
    int n,
    int k,
    const void* lhs,
    const void* rhs_int4,
    const void* scale,
    const void* zero,
    const void* awq_inv_scale,
    int group_size,
    int quant_type,
    void* out);

extern "C" void supersonic_gqh_hip_enable_tight_decode();
extern "C" int supersonic_gqh_hip_ensure_tight(
    int device_ordinal,
    int rung,
    void* wire,
    int in_dim,
    int out_dim);
extern "C" int supersonic_gqh_hip_ensure_padded(
    int device_ordinal,
    int rung,
    void* wire,
    int in_dim,
    int out_dim);

extern "C" int supersonic_gqh_hip_matvec_stream(
    int device_ordinal,
    int rung,
    const void* wire,
    const void* x,
    void* y,
    int in_dim,
    int out_dim,
    int ncols,
    int64_t x_col_stride,
    int64_t y_col_stride,
    float tensor_scale,
    int grid_code,
    void* stream);

extern "C" int supersonic_gqh_hip_matvec_stream_acc(
    int device_ordinal,
    int rung,
    const void* wire,
    const void* x,
    void* y,
    int in_dim,
    int out_dim,
    int ncols,
    int64_t x_col_stride,
    int64_t y_col_stride,
    float tensor_scale,
    int grid_code,
    void* stream);

extern "C" int supersonic_gqh_hip_mix_matvec_stream(
    int device_ordinal,
    int qtype,
    const void* wire,
    const void* x,
    void* y,
    int in_dim,
    int out_dim,
    int ncols,
    int acc,
    int mode,
    const float* lut,
    void* stream);

extern "C" int supersonic_qwen35_hip_decode_rec_k128_fused(
    int device_ordinal,
    int nv,
    int nk,
    float* rec_state,
    const float* q_unique,
    const float* k_unique,
    const float* value,
    const float* b,
    const float* a,
    const hip_bfloat16* dt_bias,
    const hip_bfloat16* a_log_exp,
    float* out,
    void* stream);

extern "C" int supersonic_qwen35_hip_delta_recurrent_prefill_on_stream(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t seq_len,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* initial_state,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out,
    void* stream);

extern "C" int supersonic_qwen35_hip_full_attention_prefill(
    int dtype,
    size_t device_ordinal,
    size_t batch_size,
    size_t q_heads,
    size_t kv_heads,
    size_t q_len,
    size_t kv_len,
    size_t head_dim,
    size_t num_kv_groups,
    float scale,
    size_t seqlen_offset,
    const void* query,
    const void* key,
    const void* value,
    void* out);

extern "C" void supersonic_gqh_hip_set_row_off(int off);

extern "C" int supersonic_gqh_hip_matvec_stream_ab(
    int device_ordinal,
    int rung_a,
    int rung_b,
    const void* wire_a,
    const void* wire_b,
    const void* x,
    void* y_a,
    void* y_b,
    int in_dim,
    int out_a,
    int out_b,
    float scale_a,
    float scale_b,
    int grid_a,
    int grid_b,
    void* stream);

extern "C" int supersonic_gqh_hip_matvec_stream_pair(
    int device_ordinal,
    int rung_a,
    int rung_b,
    const void* wire_a,
    const void* wire_b,
    const void* x,
    void* y_a,
    void* y_b,
    int in_dim,
    int out_dim,
    int out_b,
    float scale_a,
    float scale_b,
    int grid_a,
    int grid_b,
    int fuse_swiglu,
    void* stream,
    int ncols,
    int64_t x_col_stride,
    int64_t y_col_stride);

namespace {

// The GQH bridge and this translation unit share process-global HIP metadata,
// graph objects, streams, and scratch allocations. This recursive guard lets
// persistent decode serialize that state while still calling the GQH entry
// points, which take the same bridge mutex themselves.
struct DecodeBridgeLockGuard {
    DecodeBridgeLockGuard() { supersonic_gqh_hip_lock(); }
    ~DecodeBridgeLockGuard() { supersonic_gqh_hip_unlock(); }
};

// HIP allocations and streams are owned by a device ordinal, not by the
// thread's incidental current-device setting. Keep the switch status visible
// to callers that are about to touch an allocation. Explicit restore reports
// the operation; the destructor applies the same fail-stop policy if it is
// reached with an un-restored device switch.
struct ScopedHipDevice {
    int previous = -1;
    bool changed = false;
    hipError_t status = hipSuccess;

    explicit ScopedHipDevice(int target) {
        status = hipGetDevice(&previous);
        if (status != hipSuccess) {
            return;
        }
        if (previous != target) {
            status = hipSetDevice(target);
            if (status == hipSuccess) {
                changed = true;
            }
        }
    }

    hipError_t restore() {
        if (!changed || previous < 0) {
            return hipSuccess;
        }
        const hipError_t err = hipSetDevice(previous);
        if (err == hipSuccess) {
            changed = false;
        } else {
            supersonic_gpu_integrity_fail_stop(
                "4b device restore", static_cast<int>(err), previous);
        }
        return hipSuccess;
    }

    ~ScopedHipDevice() {
        const hipError_t err = restore();
        if (err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "4b device restore", static_cast<int>(err), previous);
        }
    }

    bool ok() const { return status == hipSuccess; }
};

int gqh_rung_from_qtype(int qtype) {
    if (qtype == 108) return 0;
    if (qtype == 109) return 1;
    if (qtype == 110) return 2;
    if (qtype == 111) return 3;
    return -1;
}

struct GqhProjHdr {
    const void* wire = nullptr;
    float scale = 0.0f;
    int grid = 0;
    int rung = -1;
    int qtype = 0;
    int mix_mode = 0;
    int mix_k = 0;
    float mix_lut[16] = {};
};

bool ggml_k_qtype(int qtype) {
    return qtype == 8 || qtype == 12 || qtype == 13 || qtype == 14;
}

bool mix_qtype(int qtype) {
    return qtype == 105 || qtype == 106;
}

bool proj_can_gemv(const GqhProjHdr& h) {
    return h.wire != nullptr &&
        (h.rung >= 0 || ggml_k_qtype(h.qtype) || mix_qtype(h.qtype));
}

struct GqhMixerLayer {
    int layer_type = 0;
    int q_out = 0;
    int k_out = 0;
    int attn_size = 0;
    int attn_heads = 0;
    int qkv_out = 0;
    int z_out = 0;
    int nv = 0;
    int val_dim = 0;
    int hkd = 0;
    int hvd = 0;
    float linear_norm_eps = 0.0f;
    const void* input_norm_w = nullptr;
    float input_norm_eps = 0.0f;
    const void* post_attn_norm_w = nullptr;
    float post_attn_norm_eps = 0.0f;
    int rms_unit_offset = 0;
    void* recurrent_state = nullptr;
    const void* dt_bias_w = nullptr;
    const void* a_log_exp_w = nullptr;
    const void* linear_norm_w = nullptr;
    void* kv_cache_k = nullptr;
    void* kv_cache_v = nullptr;
    int kv_len = 0;
    int kv_max_t = 0;
    void* kv_shadow_k = nullptr;
    void* kv_shadow_v = nullptr;
    int kv_shadow_start = -1;
    void* debug_linear_trace_out = nullptr;
    int debug_linear_trace_channel = -1;
    int attn_head_dim = 0;
    int attn_kv_heads = 0;
    const void* q_norm_w = nullptr;
    const void* k_norm_w = nullptr;
    void* conv_state = nullptr;
    const void* conv1d_w = nullptr;
    int conv_kernel_size = 0;
    GqhProjHdr q, k, v, o;
    GqhProjHdr qkv, z, b, a, lin_out;
};

struct GqhMlpHdrs {
    GqhProjHdr gate[80];
    GqhProjHdr up[80];
    GqhProjHdr down[80];
    GqhMixerLayer mix[80];
    const void* layers = nullptr;
    const void* int4 = nullptr;
    int device_ordinal = -1;
    int n = 0;
    uint64_t state_signature = 0;
    bool ok = false;
};

// These caches contain raw device pointers copied from one DecodeEngine. They
// are process-global for the retained single-threaded bridge, but their owner
// must explicitly invalidate them before that engine frees its buffers.
GqhMlpHdrs g_gqh_mlp_hdrs;

struct SplitGraphCache {
    hipGraphExec_t exec = nullptr;
    hipGraph_t graph = nullptr;
    hipStream_t stream = nullptr;
    int num_layers = -1;
    int device_ordinal = -1;
    int grid_in = 0, grid_mid = 0, grid_out = 0, grid_gate = 0, grid_up = 0,
        grid_down = 0;
    const void* layers = nullptr;
    void* hidden_io = nullptr;
    float* workspace = nullptr;
    unsigned int* counters = nullptr;
    unsigned int* barrier_counter = nullptr;
    unsigned int* barrier_flag = nullptr;
    const void* int4 = nullptr;
    const void* cos_table = nullptr;
    const void* sin_table = nullptr;
    const void* fp8_scales = nullptr;
    const void* kv_fp8_descs = nullptr;
    const void* batch_descs = nullptr;
    uint64_t state_signature = 0;
    int batch_size = 0;
};

SplitGraphCache g_split_graph_cache;

hipError_t load_gqh_proj_hdr(
    const void* wire, const void* scale_ptr, int qtype, GqhProjHdr* out) {
    out->wire = wire;
    out->qtype = qtype;
    out->rung = gqh_rung_from_qtype(qtype);
    out->scale = 0.0f;
    out->grid = 0;
    if (wire == nullptr) {
        out->rung = -1;
        out->qtype = 0;
        return hipSuccess;
    }
    if (mix_qtype(out->qtype)) {
        if (scale_ptr == nullptr) {
            out->qtype = 0;
            return hipSuccess;
        }
        struct MixSide {
            float lut[16];
            int mode;
            int k;
        } side{};
        hipError_t err = hipMemcpy(&side, scale_ptr, sizeof(side), hipMemcpyDeviceToHost);
        if (err != hipSuccess) {
            return err;
        }
        out->mix_mode = side.mode;
        out->mix_k = side.k;
        memcpy(out->mix_lut, side.lut, sizeof(out->mix_lut));
        return hipSuccess;
    }
    if (out->rung < 0 || scale_ptr == nullptr) {
        if (out->rung >= 0) {
            out->rung = -1;
        }
        return hipSuccess;
    }
    Qwen35GqhHdr h{};
    hipError_t err = hipMemcpy(&h, scale_ptr, sizeof(h), hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        return err;
    }
    out->scale = h.tensor_scale;
    out->grid = h.grid_code;
    return hipSuccess;
}

// Refresh the descriptor fields that are copied into host-side GEMV helpers.
// The descriptor allocation is intentionally stable across DecodeEngine
// launches, while recurrent/conv/KV buffers are growable and may be replaced
// by reset. Never retain those pointers solely because `layers_dev` is the
// same descriptor allocation.
void refresh_gqh_mixer_layer(
    const Qwen35DecodeLayerDesc& L, GqhMixerLayer* m) {
    m->layer_type = L.layer_type;
    m->q_out = L.q_out_dim;
    m->k_out = L.k_out_dim;
    m->attn_size = L.attn_num_heads * L.attn_head_dim;
    m->attn_heads = L.attn_num_heads;
    m->qkv_out = L.qkv_out_dim;
    m->z_out = L.z_out_dim;
    m->nv = L.linear_num_v_heads;
    m->val_dim = L.linear_value_dim;
    m->hkd = L.linear_head_k_dim;
    m->hvd = L.linear_head_v_dim;
    m->linear_norm_eps = L.linear_norm_eps;
    m->input_norm_w = L.input_norm_w;
    m->input_norm_eps = L.input_norm_eps;
    m->post_attn_norm_w = L.post_attn_norm_w;
    m->post_attn_norm_eps = L.post_attn_norm_eps;
    m->rms_unit_offset = L.rms_norm_add_unit_offset;
    m->recurrent_state = L.recurrent_state;
    m->dt_bias_w = L.dt_bias_w;
    m->a_log_exp_w = L.a_log_exp_w;
    m->linear_norm_w = L.linear_norm_w;
    m->kv_cache_k = L.kv_cache_k;
    m->kv_cache_v = L.kv_cache_v;
    m->kv_len = L.kv_len;
    m->kv_max_t = L.kv_max_t;
    m->kv_shadow_k = L.kv_shadow_k;
    m->kv_shadow_v = L.kv_shadow_v;
    m->kv_shadow_start = L.kv_shadow_start;
    m->attn_head_dim = L.attn_head_dim;
    m->attn_kv_heads = L.attn_num_kv_heads;
    m->q_norm_w = L.q_norm_w;
    m->k_norm_w = L.k_norm_w;
    m->conv_state = L.conv_state;
    m->conv1d_w = L.conv1d_w;
    m->conv_kernel_size = L.conv_kernel_size;
    m->debug_linear_trace_out = L.debug_linear_trace_out;
    m->debug_linear_trace_channel = L.debug_linear_trace_channel;
}

void gqh_hash_word(uint64_t* hash, uint64_t word) {
    // FNV-1a over fixed-width words keeps pointer identity and scalar state
    // fields in the graph ownership key without depending on struct padding.
    *hash ^= word;
    *hash *= 1099511628211ull;
}

void gqh_hash_ptr(uint64_t* hash, const void* ptr) {
    gqh_hash_word(hash, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr)));
}

uint64_t gqh_mlp_state_signature(const GqhMlpHdrs& cache, int num_layers) {
    uint64_t hash = 1469598103934665603ull;
    gqh_hash_word(&hash, static_cast<uint64_t>(cache.device_ordinal));
    for (int layer = 0; layer < num_layers; ++layer) {
        const GqhMixerLayer& m = cache.mix[layer];
        gqh_hash_word(&hash, static_cast<uint64_t>(m.layer_type));
        gqh_hash_word(&hash, static_cast<uint64_t>(m.q_out));
        gqh_hash_word(&hash, static_cast<uint64_t>(m.k_out));
        gqh_hash_word(&hash, static_cast<uint64_t>(m.attn_heads));
        gqh_hash_word(&hash, static_cast<uint64_t>(m.attn_kv_heads));
        gqh_hash_word(&hash, static_cast<uint64_t>(m.attn_head_dim));
        gqh_hash_word(&hash, static_cast<uint64_t>(m.qkv_out));
        gqh_hash_word(&hash, static_cast<uint64_t>(m.z_out));
        gqh_hash_word(&hash, static_cast<uint64_t>(m.nv));
        gqh_hash_word(&hash, static_cast<uint64_t>(m.val_dim));
        gqh_hash_word(&hash, static_cast<uint64_t>(m.hkd));
        gqh_hash_word(&hash, static_cast<uint64_t>(m.hvd));
        gqh_hash_ptr(&hash, m.recurrent_state);
        gqh_hash_ptr(&hash, m.conv_state);
        gqh_hash_ptr(&hash, m.kv_cache_k);
        gqh_hash_ptr(&hash, m.kv_cache_v);
        gqh_hash_word(&hash, static_cast<uint64_t>(m.kv_max_t));
        gqh_hash_ptr(&hash, m.kv_shadow_k);
        gqh_hash_ptr(&hash, m.kv_shadow_v);
        gqh_hash_word(&hash, static_cast<uint64_t>(m.kv_shadow_start));
        gqh_hash_ptr(&hash, m.debug_linear_trace_out);
        gqh_hash_word(&hash, static_cast<uint64_t>(m.debug_linear_trace_channel));
    }
    return hash;
}

bool refresh_cached_gqh_mixer_descs(
    const Qwen35DecodeLayerDesc* layers_dev,
    int num_layers,
    GqhMlpHdrs* cache) {
    for (int layer = 0; layer < num_layers; ++layer) {
        Qwen35DecodeLayerDesc L{};
        if (hipMemcpy(&L, layers_dev + layer, sizeof(L), hipMemcpyDeviceToHost) !=
            hipSuccess) {
            return false;
        }
        refresh_gqh_mixer_layer(L, &cache->mix[layer]);
    }
    cache->state_signature = gqh_mlp_state_signature(*cache, num_layers);
    return true;
}

bool load_gqh_mlp_hdrs(
    int device_ordinal,
    const Qwen35DecodeLayerDesc* layers_dev,
    const Qwen35INT4ScaleDesc* int4_dev,
    int num_layers,
    GqhMlpHdrs* cache) {
    if (cache->ok && cache->device_ordinal == device_ordinal &&
        cache->layers == layers_dev && cache->int4 == int4_dev &&
        cache->n == num_layers) {
        cache->ok = refresh_cached_gqh_mixer_descs(layers_dev, num_layers, cache);
        return cache->ok;
    }
    if (layers_dev == nullptr || int4_dev == nullptr || num_layers <= 0 ||
        num_layers > 80) {
        return false;
    }
    for (int layer = 0; layer < num_layers; ++layer) {
        Qwen35DecodeLayerDesc L{};
        Qwen35INT4ScaleDesc S{};
        if (hipMemcpy(&L, layers_dev + layer, sizeof(L), hipMemcpyDeviceToHost) !=
            hipSuccess) {
            return false;
        }
        if (hipMemcpy(&S, int4_dev + layer, sizeof(S), hipMemcpyDeviceToHost) !=
            hipSuccess) {
            return false;
        }
        if (load_gqh_proj_hdr(L.gate_proj_w, S.gate_proj_scale, S.gate_proj_type,
                              &cache->gate[layer]) != hipSuccess) {
            return false;
        }
        if (load_gqh_proj_hdr(L.up_proj_w, S.up_proj_scale, S.up_proj_type,
                              &cache->up[layer]) != hipSuccess) {
            return false;
        }
        if (load_gqh_proj_hdr(L.down_proj_w, S.down_proj_scale, S.down_proj_type,
                              &cache->down[layer]) != hipSuccess) {
            return false;
        }
        GqhMixerLayer& m = cache->mix[layer];
        refresh_gqh_mixer_layer(L, &m);
        if (load_gqh_proj_hdr(L.q_proj_w, S.q_proj_scale, S.q_proj_type, &m.q) !=
                hipSuccess ||
            load_gqh_proj_hdr(L.k_proj_w, S.k_proj_scale, S.k_proj_type, &m.k) !=
                hipSuccess ||
            load_gqh_proj_hdr(L.v_proj_w, S.v_proj_scale, S.v_proj_type, &m.v) !=
                hipSuccess ||
            load_gqh_proj_hdr(L.o_proj_w, S.o_proj_scale, S.o_proj_type, &m.o) !=
                hipSuccess ||
            load_gqh_proj_hdr(L.qkv_proj_w, S.qkv_proj_scale, S.qkv_proj_type,
                              &m.qkv) != hipSuccess ||
            load_gqh_proj_hdr(L.z_proj_w, S.z_proj_scale, S.z_proj_type, &m.z) !=
                hipSuccess ||
            load_gqh_proj_hdr(L.b_proj_w, nullptr, S.b_proj_type, &m.b) !=
                hipSuccess ||
            load_gqh_proj_hdr(L.a_proj_w, nullptr, S.a_proj_type, &m.a) !=
                hipSuccess ||
            load_gqh_proj_hdr(
                L.linear_out_proj_w,
                S.linear_out_proj_scale,
                S.linear_out_proj_type,
                &m.lin_out) != hipSuccess) {
            return false;
        }
        m.b.wire = L.b_proj_w;
        m.a.wire = L.a_proj_w;
    }
    cache->layers = layers_dev;
    cache->int4 = int4_dev;
    cache->device_ordinal = device_ordinal;
    cache->n = num_layers;
    cache->state_signature = gqh_mlp_state_signature(*cache, num_layers);
    cache->ok = true;
    int lin_n = 0, lin_gqh = 0, lin_ggml = 0, full_n = 0, full_gqh = 0,
        full_ggml = 0, full_kv = 0, lin_ab = 0;
    int mlp_pair = 0, mlp_down_gqh = 0, mlp_down_ggml = 0, mlp_down_mix = 0;
    int ggml_qt[4] = {0, 0, 0, 0};
    auto bump_ggml = [&](int qt) {
        if (qt == 8) {
            ++ggml_qt[0];
        } else if (qt == 12) {
            ++ggml_qt[1];
        } else if (qt == 13) {
            ++ggml_qt[2];
        } else if (qt == 14) {
            ++ggml_qt[3];
        }
    };
    for (int li = 0; li < num_layers; ++li) {
        const GqhMixerLayer& m = cache->mix[li];
        if (m.layer_type == 1) {
            ++full_n;
            if (m.o.rung >= 0) {
                ++full_gqh;
            } else if (ggml_k_qtype(m.o.qtype)) {
                ++full_ggml;
            }
            if (proj_can_gemv(m.k) && proj_can_gemv(m.v)) {
                ++full_kv;
            }
            bump_ggml(m.k.qtype);
            bump_ggml(m.v.qtype);
            if (m.o.rung < 0) {
                bump_ggml(m.o.qtype);
            }
        } else {
            ++lin_n;
            if (m.lin_out.rung >= 0) {
                ++lin_gqh;
            } else if (ggml_k_qtype(m.lin_out.qtype)) {
                ++lin_ggml;
            } else if (mix_qtype(m.lin_out.qtype)) {
                std::fprintf(
                    stderr,
                    "[gqh-gemv] mix lin L%d qt=%d mode=%d k=%d lut0=%.6g lut7=%.6g\n",
                    li,
                    m.lin_out.qtype,
                    m.lin_out.mix_mode,
                    m.lin_out.mix_k,
                    m.lin_out.mix_lut[0],
                    m.lin_out.mix_lut[7]);
            }
            if (mix_qtype(cache->down[li].qtype)) {
                std::fprintf(
                    stderr,
                    "[gqh-gemv] mix down L%d qt=%d mode=%d k=%d lut0=%.6g lut3=%.6g\n",
                    li,
                    cache->down[li].qtype,
                    cache->down[li].mix_mode,
                    cache->down[li].mix_k,
                    cache->down[li].mix_lut[0],
                    cache->down[li].mix_lut[3]);
            }
            if (proj_can_gemv(m.a) && proj_can_gemv(m.b)) {
                ++lin_ab;
            }
            bump_ggml(m.a.qtype);
            bump_ggml(m.b.qtype);
            if (m.lin_out.rung < 0) {
                bump_ggml(m.lin_out.qtype);
            }
        }
        if (cache->gate[li].rung >= 0 && cache->up[li].rung >= 0) {
            ++mlp_pair;
        }
        if (cache->down[li].rung >= 0) {
            ++mlp_down_gqh;
        } else if (ggml_k_qtype(cache->down[li].qtype)) {
            ++mlp_down_ggml;
        } else if (mix_qtype(cache->down[li].qtype)) {
            ++mlp_down_mix;
        }
        if (li < 4) {
            std::fprintf(
                stderr,
                "[gqh-gemv] L%d type=%d q=%d k=%d attn=%d qkv=%d z=%d nv=%d vd=%d "
                "q_rung=%d k_rung=%d v_rung=%d o_rung=%d qkv_rung=%d z_rung=%d "
                "lin_rung=%d k_qt=%d v_qt=%d a_qt=%d b_qt=%d o_qt=%d lin_qt=%d\n",
                li,
                m.layer_type,
                m.q_out,
                m.k_out,
                m.attn_size,
                m.qkv_out,
                m.z_out,
                m.nv,
                m.val_dim,
                m.q.rung,
                m.k.rung,
                m.v.rung,
                m.o.rung,
                m.qkv.rung,
                m.z.rung,
                m.lin_out.rung,
                m.k.qtype,
                m.v.qtype,
                m.a.qtype,
                m.b.qtype,
                m.o.qtype,
                m.lin_out.qtype);
        }
    }
    std::fprintf(
        stderr,
        "[gqh-gemv] mlp pair=%d/%d gqh  down=%d gqh + %d ggml-K + %d mix\n",
        mlp_pair,
        num_layers,
        mlp_down_gqh,
        mlp_down_ggml,
        mlp_down_mix);
    std::fprintf(
        stderr,
        "[gqh-gemv] out rungs lin=%d/%d gqh + %d ggml-K  full=%d/%d gqh + %d ggml-K  "
        "in kv=%d/%d ab=%d/%d  ggml-K q8=%d q4k=%d q5k=%d q6k=%d\n",
        lin_gqh,
        lin_n,
        lin_ggml,
        full_gqh,
        full_n,
        full_ggml,
        full_kv,
        full_n,
        lin_ab,
        lin_n,
        ggml_qt[0],
        ggml_qt[1],
        ggml_qt[2],
        ggml_qt[3]);
    return true;
}

// Side stream so rec qkv and z GEMVs can overlap. Same kernels as the
// sequential path; only the launch stream differs. The ordinal is part of
// ownership because HIP streams/events are device-local.
struct DecodeSideResources {
    int device_ordinal = -1;
    hipStream_t stream = nullptr;
    hipEvent_t events[2] = {nullptr, nullptr};
};

DecodeSideResources& decode_side_resources() {
    static DecodeSideResources s;
    return s;
}

hipError_t reset_decode_side_resources(DecodeSideResources& s) {
    const int owner = s.device_ordinal;
    if (owner < 0) {
        if (s.stream != nullptr || s.events[0] != nullptr || s.events[1] != nullptr) {
            supersonic_gpu_integrity_fail_stop(
                "decode side resources missing owner", static_cast<int>(hipErrorInvalidDevice), owner);
        }
        return hipSuccess;
    }
    ScopedHipDevice scoped(owner);
    if (!scoped.ok()) {
        supersonic_gpu_integrity_fail_stop(
            "decode side resource owner switch", static_cast<int>(scoped.status), owner);
    }
    if (s.stream != nullptr || s.events[0] != nullptr || s.events[1] != nullptr) {
        const hipError_t sync_err = hipDeviceSynchronize();
        if (sync_err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "decode side resource synchronize", static_cast<int>(sync_err), owner);
        }
    }
    for (hipEvent_t& event : s.events) {
        if (event != nullptr) {
            const hipError_t err = hipEventDestroy(event);
            if (err != hipSuccess) {
                supersonic_gpu_integrity_fail_stop(
                    "decode side event destroy", static_cast<int>(err), owner);
            }
            event = nullptr;
        }
    }
    if (s.stream != nullptr) {
        const hipError_t err = hipStreamDestroy(s.stream);
        if (err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "decode side stream destroy", static_cast<int>(err), owner);
        }
        s.stream = nullptr;
    }
    s.device_ordinal = -1;
    const hipError_t restore_err = scoped.restore();
    if (restore_err != hipSuccess) {
        supersonic_gpu_integrity_fail_stop(
            "decode side resource owner restore", static_cast<int>(restore_err), owner);
    }
    return hipSuccess;
}

hipError_t ensure_decode_side_resources(int ordinal) {
    DecodeSideResources& s = decode_side_resources();
    if (s.device_ordinal >= 0 && s.device_ordinal != ordinal) {
        const hipError_t err = reset_decode_side_resources(s);
        if (err != hipSuccess) {
            return err;
        }
    }
    if (s.device_ordinal < 0) {
        s.device_ordinal = ordinal;
    }
    if (s.stream == nullptr) {
        const hipError_t err = hipStreamCreateWithFlags(&s.stream, hipStreamNonBlocking);
        if (err != hipSuccess) {
            const hipError_t reset_err = reset_decode_side_resources(s);
            return reset_err != hipSuccess ? reset_err : err;
        }
    }
    for (hipEvent_t& event : s.events) {
        if (event == nullptr) {
            const hipError_t err = hipEventCreateWithFlags(&event, hipEventDisableTiming);
            if (err != hipSuccess) {
                const hipError_t reset_err = reset_decode_side_resources(s);
                return reset_err != hipSuccess ? reset_err : err;
            }
        }
    }
    return hipSuccess;
}

hipError_t decode_side_stream(int ordinal, hipStream_t* out) {
    if (out == nullptr) {
        return hipErrorInvalidValue;
    }
    const hipError_t err = ensure_decode_side_resources(ordinal);
    if (err != hipSuccess) {
        return err;
    }
    *out = decode_side_resources().stream;
    return hipSuccess;
}

hipError_t decode_fork_side(int ordinal, hipStream_t main, hipStream_t side) {
    if (side == nullptr) {
        return hipSuccess;
    }
    const hipError_t ensure_err = ensure_decode_side_resources(ordinal);
    if (ensure_err != hipSuccess) {
        return ensure_err;
    }
    hipEvent_t ev = decode_side_resources().events[0];
    if (ev == nullptr) {
        return hipErrorInvalidResourceHandle;
    }
    const hipError_t record_err = hipEventRecord(ev, main);
    if (record_err != hipSuccess) {
        supersonic_gpu_integrity_fail_stop(
            "decode side fork event record", static_cast<int>(record_err), ordinal);
    }
    const hipError_t wait_err = hipStreamWaitEvent(side, ev, 0);
    if (wait_err != hipSuccess) {
        supersonic_gpu_integrity_fail_stop(
            "decode side fork event wait", static_cast<int>(wait_err), ordinal);
    }
    return hipSuccess;
}

hipError_t decode_join_side(int ordinal, hipStream_t main, hipStream_t side) {
    if (side == nullptr) {
        return hipSuccess;
    }
    const hipError_t ensure_err = ensure_decode_side_resources(ordinal);
    if (ensure_err != hipSuccess) {
        return ensure_err;
    }
    hipEvent_t ev = decode_side_resources().events[1];
    if (ev == nullptr) {
        return hipErrorInvalidResourceHandle;
    }
    const hipError_t record_err = hipEventRecord(ev, side);
    if (record_err != hipSuccess) {
        supersonic_gpu_integrity_fail_stop(
            "decode side join event record", static_cast<int>(record_err), ordinal);
    }
    const hipError_t wait_err = hipStreamWaitEvent(main, ev, 0);
    if (wait_err != hipSuccess) {
        supersonic_gpu_integrity_fail_stop(
            "decode side join event wait", static_cast<int>(wait_err), ordinal);
    }
    return hipSuccess;
}

// A side-stream GEMV may be deliberately left running while the main stream
// prepares recurrent state. Once it is recorded in this guard, every control
// flow edge out of the layer joins it. This is deliberately fail-stop on a
// join-resource error: returning to Rust with z still in flight would let the
// owning workspace be freed while the side stream still reads it.
struct DecodeSideJoinGuard {
    int ordinal;
    hipStream_t main;
    hipStream_t pending = nullptr;

    DecodeSideJoinGuard(int ordinal_, hipStream_t main_)
        : ordinal(ordinal_), main(main_) {}

    void defer(hipStream_t side) {
        pending = side;
    }

    bool active() const {
        return pending != nullptr;
    }

    hipError_t join() {
        if (pending == nullptr) {
            return hipSuccess;
        }
        const hipStream_t side = pending;
        pending = nullptr;
        const hipError_t err = decode_join_side(ordinal, main, side);
        if (err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "decode side join guard", static_cast<int>(err), ordinal);
        }
        return hipSuccess;
    }

    ~DecodeSideJoinGuard() {
        if (pending == nullptr) {
            return;
        }
        const hipStream_t side = pending;
        pending = nullptr;
        const hipError_t err = decode_join_side(ordinal, main, side);
        if (err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "decode side join guard", static_cast<int>(err), ordinal);
        }
    }
};

hipError_t launch_gqh_gemv(
    int ordinal,
    const GqhProjHdr& h,
    const float* x,
    float* y,
    int in_dim,
    int out_dim,
    hipStream_t stream,
    int ncols = 1,
    int64_t x_col_stride = 0,
    int64_t y_col_stride = 0) {
    if (ncols <= 0) {
        ncols = 1;
    }
    if (x_col_stride <= 0) {
        x_col_stride = in_dim;
    }
    if (y_col_stride <= 0) {
        y_col_stride = out_dim;
    }
    const int st = supersonic_gqh_hip_matvec_stream(
        ordinal,
        h.rung,
        h.wire,
        x,
        y,
        in_dim,
        out_dim,
        ncols,
        x_col_stride,
        y_col_stride,
        h.scale,
        h.grid,
        stream);
    return st == 0 ? hipSuccess : hipErrorInvalidValue;
}

hipError_t launch_gqh_gemv_pair(
    int ordinal,
    const GqhProjHdr& a,
    const GqhProjHdr& b,
    const float* x,
    float* y_a,
    float* y_b,
    int in_dim,
    int out_a,
    int out_b,
    hipStream_t stream,
    bool fuse_swiglu = false,
    int ncols = 1,
    int64_t x_col_stride = 0,
    int64_t y_col_stride = 0) {
    if (a.rung < 0 || b.rung < 0 || a.wire == nullptr || b.wire == nullptr) {
        return hipErrorInvalidValue;
    }
    if (ncols <= 0) {
        ncols = 1;
    }
    if (x_col_stride <= 0) {
        x_col_stride = in_dim;
    }
    if (y_col_stride <= 0) {
        y_col_stride = out_a > 0 ? out_a : in_dim;
    }
    const int st = supersonic_gqh_hip_matvec_stream_pair(
        ordinal,
        a.rung,
        b.rung,
        a.wire,
        b.wire,
        x,
        y_a,
        y_b,
        in_dim,
        out_a,
        out_b,
        a.scale,
        b.scale,
        a.grid,
        b.grid,
        fuse_swiglu ? 1 : 0,
        stream,
        ncols,
        x_col_stride,
        y_col_stride);
    return st == 0 ? hipSuccess : hipErrorInvalidValue;
}

hipError_t launch_gqh_gemv_ab(
    int ordinal,
    const GqhProjHdr& a,
    const GqhProjHdr& b,
    const float* x,
    float* y_a,
    float* y_b,
    int in_dim,
    int out_a,
    int out_b,
    hipStream_t stream) {
    if (a.rung < 0 || b.rung < 0 || a.wire == nullptr || b.wire == nullptr) {
        return hipErrorInvalidValue;
    }
    const int st = supersonic_gqh_hip_matvec_stream_ab(
        ordinal,
        a.rung,
        b.rung,
        a.wire,
        b.wire,
        x,
        y_a,
        y_b,
        in_dim,
        out_a,
        out_b,
        a.scale,
        b.scale,
        a.grid,
        b.grid,
        stream);
    return st == 0 ? hipSuccess : hipErrorInvalidValue;
}

hipError_t launch_gqh_gemv_acc(
    int ordinal,
    const GqhProjHdr& h,
    const float* x,
    float* y,
    int in_dim,
    int out_dim,
    hipStream_t stream,
    int ncols = 1,
    int64_t x_col_stride = 0,
    int64_t y_col_stride = 0) {
    if (ncols <= 0) {
        ncols = 1;
    }
    if (x_col_stride <= 0) {
        x_col_stride = in_dim;
    }
    if (y_col_stride <= 0) {
        y_col_stride = out_dim;
    }
    const int st = supersonic_gqh_hip_matvec_stream_acc(
        ordinal,
        h.rung,
        h.wire,
        x,
        y,
        in_dim,
        out_dim,
        ncols,
        x_col_stride,
        y_col_stride,
        h.scale,
        h.grid,
        stream);
    return st == 0 ? hipSuccess : hipErrorInvalidValue;
}

void launch_round_f32(
    float* y,
    int n,
    hipStream_t stream,
    int ncols = 1,
    int64_t col_stride = 0) {
    if (ncols <= 0) {
        ncols = 1;
    }
    const int64_t stride = col_stride > 0 ? col_stride : static_cast<int64_t>(n);
    if (ncols == 1 || stride == static_cast<int64_t>(n)) {
        const int total = n * ncols;
        const int bs = 256;
        const int gs = (total + bs - 1) / bs;
        hipLaunchKernelGGL(
            supersonic_qwen35_bf16_round_f32_kernel,
            dim3(static_cast<unsigned int>(gs)),
            dim3(bs),
            0,
            stream,
            y,
            total);
        return;
    }
    const int bs = 256;
    const int gs = (n + bs - 1) / bs;
    for (int c = 0; c < ncols; ++c) {
        hipLaunchKernelGGL(
            supersonic_qwen35_bf16_round_f32_kernel,
            dim3(static_cast<unsigned int>(gs)),
            dim3(bs),
            0,
            stream,
            y + static_cast<int64_t>(c) * stride,
            n);
    }
}

void launch_add_round_f32(float* dst, const float* add, int n, hipStream_t stream) {
    const int bs = 256;
    const int gs = (n + bs - 1) / bs;
    hipLaunchKernelGGL(
        supersonic_qwen35_add_round_f32_kernel,
        dim3(static_cast<unsigned int>(gs)),
        dim3(bs),
        0,
        stream,
        dst,
        add,
        n);
}

struct DecodeRecScratch {
    float* q = nullptr;
    float* k = nullptr;
    float* beta = nullptr;
    float* g = nullptr;
    float* out = nullptr;
    int device_ordinal = -1;
    size_t cap = 0;
};

DecodeRecScratch& decode_rec_scratch() {
    static DecodeRecScratch s;
    return s;
}

// Per-token linear (conv+rec) snapshots for fused B>1 MTP verify. After a
// partial accept we restore prefix `commit_len` instead of replaying the
// trunk sequentially. Rec is copied after each token in the layer B-loop;
// conv is written inside the rec-prep kernel after each col.
struct MtpPrefixSnap {
    static constexpr int kMaxB = 8;
    static constexpr int kMaxLayers = 80;
    float* rec[kMaxB][kMaxLayers]{};
    hip_bfloat16* conv[kMaxB][kMaxLayers]{};
    void* rec_live[kMaxLayers]{};
    void* conv_live[kMaxLayers]{};
    size_t rec_bytes[kMaxLayers]{};
    size_t conv_bytes[kMaxLayers]{};
    int n_layers = 0;
    int n_b = 0;
    bool ready = false;
    float* rec_slab = nullptr;
    hip_bfloat16* conv_slab = nullptr;
    size_t rec_slab_bytes = 0;
    size_t conv_slab_bytes = 0;
    int device_ordinal = -1;
    const void* owner_layers = nullptr;
};

MtpPrefixSnap& mtp_prefix_snap() {
    static MtpPrefixSnap s;
    return s;
}

hipError_t release_mtp_prefix_slabs(
    MtpPrefixSnap& s, bool release_rec, bool release_conv) {
    if (s.rec_slab == nullptr && s.conv_slab == nullptr) {
        return hipSuccess;
    }
    if (s.device_ordinal < 0) {
        supersonic_gpu_integrity_fail_stop(
            "MTP prefix slabs missing owner", static_cast<int>(hipErrorInvalidDevice), -1);
    }
    ScopedHipDevice owner(s.device_ordinal);
    if (!owner.ok()) {
        supersonic_gpu_integrity_fail_stop(
            "MTP prefix slab owner switch",
            static_cast<int>(owner.status),
            s.device_ordinal);
    }
    const hipError_t sync_err = hipDeviceSynchronize();
    if (sync_err != hipSuccess) {
        supersonic_gpu_integrity_fail_stop(
            "MTP prefix slab synchronize", static_cast<int>(sync_err), s.device_ordinal);
    }
    if (release_rec && s.rec_slab != nullptr) {
        const hipError_t err = hipFree(s.rec_slab);
        if (err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "MTP prefix rec slab free", static_cast<int>(err), s.device_ordinal);
        }
        s.rec_slab = nullptr;
        s.rec_slab_bytes = 0;
    }
    if (release_conv && s.conv_slab != nullptr) {
        const hipError_t err = hipFree(s.conv_slab);
        if (err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "MTP prefix conv slab free", static_cast<int>(err), s.device_ordinal);
        }
        s.conv_slab = nullptr;
        s.conv_slab_bytes = 0;
    }
    s.ready = false;
    const hipError_t restore_err = owner.restore();
    if (restore_err != hipSuccess) {
        supersonic_gpu_integrity_fail_stop(
            "MTP prefix slab owner restore", static_cast<int>(restore_err), s.device_ordinal);
    }
    return hipSuccess;
}

hipError_t ensure_mtp_prefix_snap(
    const GqhMlpHdrs& hdrs, int n_layers, int n_b) {
    MtpPrefixSnap& s = mtp_prefix_snap();

    if (n_layers <= 0 || n_layers > MtpPrefixSnap::kMaxLayers || n_b < 2 ||
        n_b > MtpPrefixSnap::kMaxB) {
        s.ready = false;
        return hipErrorInvalidValue;
    }

    if (s.device_ordinal >= 0 && s.device_ordinal != hdrs.device_ordinal) {
        const hipError_t err = release_mtp_prefix_slabs(s, true, true);
        if (err != hipSuccess) {
            return err;
        }
        s = MtpPrefixSnap{};
    }

    size_t rec_need = 0;
    size_t conv_need = 0;
    size_t rec_bytes[MtpPrefixSnap::kMaxLayers]{};
    size_t conv_bytes[MtpPrefixSnap::kMaxLayers]{};
    for (int layer = 0; layer < n_layers; ++layer) {
        const GqhMixerLayer& mx = hdrs.mix[layer];
        if (mx.layer_type != 0 || mx.recurrent_state == nullptr ||
            mx.conv_state == nullptr || mx.nv <= 0 || mx.hkd <= 0 ||
            mx.hvd <= 0 || mx.qkv_out <= 0) {
            continue;
        }
        rec_bytes[layer] = static_cast<size_t>(mx.nv) *
            static_cast<size_t>(mx.hkd) * static_cast<size_t>(mx.hvd) *
            sizeof(float);
        conv_bytes[layer] = static_cast<size_t>(mx.qkv_out) * 3 *
            sizeof(hip_bfloat16);
        rec_need += rec_bytes[layer];
        conv_need += conv_bytes[layer];
    }
    rec_need *= static_cast<size_t>(n_b);
    conv_need *= static_cast<size_t>(n_b);
    if (rec_need == 0 || conv_need == 0) {
        s.ready = false;
        return hipSuccess;
    }
    const bool grow_rec = s.rec_slab == nullptr || s.rec_slab_bytes < rec_need;
    const bool grow_conv = s.conv_slab == nullptr || s.conv_slab_bytes < conv_need;
    // Prefix snapshots are copied by the decode stream and consumed by the
    // restore entry point after a verify round.  A capacity increase replaces
    // those slabs, so synchronize the owning device before freeing either
    // allocation even when the ordinal is unchanged.
    if ((grow_rec || grow_conv) &&
        (s.rec_slab != nullptr || s.conv_slab != nullptr)) {
        const hipError_t err = release_mtp_prefix_slabs(s, grow_rec, grow_conv);
        if (err != hipSuccess) {
            return err;
        }
    }

    ScopedHipDevice target(hdrs.device_ordinal);
    if (!target.ok()) {
        return target.status;
    }
    auto finish = [&](hipError_t err) {
        const hipError_t restore_err = target.restore();
        if (restore_err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "MTP prefix target restore", static_cast<int>(restore_err), hdrs.device_ordinal);
        }
        return err;
    };
    s.device_ordinal = hdrs.device_ordinal;
    s.owner_layers = hdrs.layers;
    if (grow_rec) {
        float* p = nullptr;
        const hipError_t err = hipMalloc(&p, rec_need);
        if (err != hipSuccess) {
            s.ready = false;
            return finish(err);
        }
        s.rec_slab = p;
        s.rec_slab_bytes = rec_need;
    }
    if (grow_conv) {
        hip_bfloat16* p = nullptr;
        const hipError_t err = hipMalloc(&p, conv_need);
        if (err != hipSuccess) {
            s.ready = false;
            return finish(err);
        }
        s.conv_slab = p;
        s.conv_slab_bytes = conv_need;
    }
    for (int layer = 0; layer < MtpPrefixSnap::kMaxLayers; ++layer) {
        for (int b = 0; b < MtpPrefixSnap::kMaxB; ++b) {
            s.rec[b][layer] = nullptr;
            s.conv[b][layer] = nullptr;
        }
        s.rec_live[layer] = nullptr;
        s.conv_live[layer] = nullptr;
        s.rec_bytes[layer] = 0;
        s.conv_bytes[layer] = 0;
    }
    char* rec_cur = reinterpret_cast<char*>(s.rec_slab);
    char* conv_cur = reinterpret_cast<char*>(s.conv_slab);
    for (int layer = 0; layer < n_layers; ++layer) {
        const GqhMixerLayer& mx = hdrs.mix[layer];
        s.rec_live[layer] = mx.recurrent_state;
        s.conv_live[layer] = mx.conv_state;
        s.rec_bytes[layer] = rec_bytes[layer];
        s.conv_bytes[layer] = conv_bytes[layer];
        if (rec_bytes[layer] == 0) {
            continue;
        }
        for (int b = 0; b < n_b; ++b) {
            s.rec[b][layer] = reinterpret_cast<float*>(rec_cur);
            rec_cur += rec_bytes[layer];
            s.conv[b][layer] = reinterpret_cast<hip_bfloat16*>(conv_cur);
            conv_cur += conv_bytes[layer];
        }
    }
    s.n_layers = n_layers;
    s.n_b = n_b;
    s.ready = true;
    return finish(hipSuccess);
}

hipError_t ensure_decode_rec_scratch(int ordinal, int nv, int khd, int vhd) {
    if (nv <= 0 || khd <= 0 || vhd <= 0) {
        return hipErrorInvalidValue;
    }
    const size_t qk = static_cast<size_t>(nv) * static_cast<size_t>(khd);
    const size_t rec_out =
        static_cast<size_t>(nv) * static_cast<size_t>(1 + khd) *
        static_cast<size_t>(vhd);
    const size_t need = qk + qk + static_cast<size_t>(nv) +
        static_cast<size_t>(nv) + rec_out;
    DecodeRecScratch& s = decode_rec_scratch();
    if (s.device_ordinal == ordinal && s.cap >= need && s.q != nullptr) {
        return hipSuccess;
    }
    if (s.q != nullptr) {
        if (s.device_ordinal < 0) {
            supersonic_gpu_integrity_fail_stop(
                "decode recurrent scratch missing owner",
                static_cast<int>(hipErrorInvalidDevice),
                -1);
        }
        const int old_device = s.device_ordinal;
        ScopedHipDevice old_owner(old_device);
        if (!old_owner.ok()) {
            supersonic_gpu_integrity_fail_stop(
                "decode recurrent scratch owner switch",
                static_cast<int>(old_owner.status),
                old_device);
        }
        hipError_t err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "decode recurrent scratch synchronize", static_cast<int>(err), old_device);
        }
        err = hipFree(s.q);
        if (err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "decode recurrent scratch free", static_cast<int>(err), old_device);
        }
        s = DecodeRecScratch{};
        err = old_owner.restore();
        if (err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "decode recurrent scratch owner restore", static_cast<int>(err), old_device);
        }
    }
    ScopedHipDevice target(ordinal);
    if (!target.ok()) {
        return target.status;
    }
    float* base = nullptr;
    hipError_t err = hipMalloc(&base, need * sizeof(float));
    if (err != hipSuccess) {
        const hipError_t restore_err = target.restore();
        if (restore_err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "decode recurrent scratch target restore",
                static_cast<int>(restore_err),
                ordinal);
        }
        return err;
    }
    s.q = base;
    s.k = s.q + qk;
    s.beta = s.k + qk;
    s.g = s.beta + nv;
    s.out = s.g + nv;
    s.device_ordinal = ordinal;
    s.cap = need;
    const hipError_t restore_err = target.restore();
    if (restore_err != hipSuccess) {
        supersonic_gpu_integrity_fail_stop(
            "decode recurrent scratch target restore", static_cast<int>(restore_err), ordinal);
    }
    return hipSuccess;
}

void launch_decode_pack_rec_inputs(
    int nv,
    int nk,
    int khd,
    const float* q_unique,
    const float* k_unique,
    const float* b,
    const float* a,
    const hip_bfloat16* dt_bias,
    const hip_bfloat16* a_log_exp,
    float* q_rep,
    float* k_rep,
    float* beta,
    float* g,
    hipStream_t stream) {
    const int n = nv * khd;
    const int bs = 256;
    const int gs = (n + bs - 1) / bs;
    hipLaunchKernelGGL(
        supersonic_qwen35_decode_pack_rec_inputs_kernel,
        dim3(static_cast<unsigned int>(gs > 0 ? gs : 1)),
        dim3(bs),
        0,
        stream,
        nv,
        nk,
        khd,
        q_unique,
        k_unique,
        b,
        a,
        dt_bias,
        a_log_exp,
        q_rep,
        k_rep,
        beta,
        g);
}

void launch_q8_0_wmma_gemv(
    const void* w,
    const float* x,
    float* y,
    int k,
    int n,
    hipStream_t stream,
    int ncols = 1,
    int64_t x_col_stride = 0,
    int64_t y_col_stride = 0) {
    if (w == nullptr || n <= 0 || k <= 0) {
        return;
    }
    if (ncols <= 0) {
        ncols = 1;
    }
    if (x_col_stride <= 0) {
        x_col_stride = k;
    }
    if (y_col_stride <= 0) {
        y_col_stride = n;
    }
    const unsigned int blocks_x =
        static_cast<unsigned int>((n + 15) / 16);
    hipLaunchKernelGGL(
        supersonic_qwen35_q8_0_wmma_gemv_f32_kernel,
        dim3(blocks_x == 0 ? 1u : blocks_x,
             static_cast<unsigned int>(ncols)),
        dim3(32),
        0,
        stream,
        static_cast<const uint8_t*>(w),
        x,
        y,
        k,
        n,
        ncols,
        x_col_stride,
        y_col_stride);
}

hipError_t launch_decode_full_prep(
    const GqhMixerLayer& mx,
    float* ws_proj,
    float* ws_attn,
    const void* cos_table,
    const void* sin_table,
    int seq_off,
    int batch_size,
    int proj_buf_floats,
    int attn_scratch_floats,
    hipStream_t stream) {
    const int nh = mx.attn_heads;
    const int nkv = mx.attn_kv_heads;
    const int hd = mx.attn_head_dim;
    if (nh <= 0 || nkv <= 0 || hd <= 0 || mx.kv_cache_k == nullptr ||
        mx.kv_cache_v == nullptr || mx.q_norm_w == nullptr ||
        mx.k_norm_w == nullptr) {
        return hipErrorInvalidValue;
    }
    if (batch_size <= 0) {
        batch_size = 1;
    }
    const int rot_dim = hd / 4;
    hipLaunchKernelGGL(
        supersonic_qwen35_decode_full_prep_kernel,
        dim3(static_cast<unsigned int>(nh),
             static_cast<unsigned int>(batch_size)),
        dim3(static_cast<unsigned int>(hd)),
        256 * sizeof(float),
        stream,
        ws_proj,
        ws_attn,
        static_cast<hip_bfloat16*>(mx.kv_cache_k),
        static_cast<hip_bfloat16*>(mx.kv_cache_v),
        static_cast<const hip_bfloat16*>(mx.q_norm_w),
        static_cast<const hip_bfloat16*>(mx.k_norm_w),
        static_cast<const hip_bfloat16*>(cos_table),
        static_cast<const hip_bfloat16*>(sin_table),
        nh,
        nkv,
        hd,
        mx.q_out,
        mx.k_out,
        rot_dim,
        mx.kv_max_t,
        seq_off,
        proj_buf_floats,
        attn_scratch_floats,
        mx.rms_unit_offset ? 1.0f : 0.0f);
    return hipGetLastError();
}

void launch_decode_rec_prep(
    int conv_dim,
    int nk,
    int hkd,
    const float* qkv_f32,
    hip_bfloat16* conv_state,
    const hip_bfloat16* conv_w,
    float* conv_out,
    hipStream_t stream,
    int ncols = 1,
    int64_t qkv_col_stride = 0,
    int64_t out_col_stride = 0,
    hip_bfloat16* conv_snaps = nullptr,
    int64_t snap_col_stride = 0) {
    if (ncols <= 0) {
        ncols = 1;
    }
    if (qkv_col_stride <= 0) {
        qkv_col_stride = conv_dim;
    }
    if (out_col_stride <= 0) {
        out_col_stride = conv_dim;
    }
    const int key_dim = nk * hkd;
    const unsigned int conv_blocks =
        static_cast<unsigned int>((conv_dim + 255) / 256);
    hipLaunchKernelGGL(
        supersonic_qwen35_decode_rec_conv_kernel,
        dim3(conv_blocks == 0 ? 1u : conv_blocks),
        dim3(256),
        0,
        stream,
        conv_dim,
        qkv_f32,
        conv_state,
        conv_w,
        conv_out,
        ncols,
        qkv_col_stride,
        out_col_stride,
        conv_snaps,
        snap_col_stride);
    hipLaunchKernelGGL(
        supersonic_qwen35_decode_rec_qk_rms_kernel,
        dim3(static_cast<unsigned int>(2 * nk),
             static_cast<unsigned int>(ncols)),
        dim3(static_cast<unsigned int>(hkd)),
        0,
        stream,
        nk,
        hkd,
        key_dim,
        conv_dim,
        conv_out,
        out_col_stride);
}

hipError_t launch_delta_recurrent_prefill_f32(
    int ordinal,
    int nv,
    int khd,
    int vhd,
    const float* state,
    const float* q,
    const float* k,
    const float* v,
    const float* beta,
    const float* g,
    float* out,
    hipStream_t stream) {
    const int st = supersonic_qwen35_hip_delta_recurrent_prefill_on_stream(
        1,
        static_cast<size_t>(ordinal),
        static_cast<size_t>(nv),
        1,
        static_cast<size_t>(khd),
        static_cast<size_t>(vhd),
        state,
        q,
        k,
        v,
        beta,
        g,
        out,
        stream);
    return st == 0 ? hipSuccess : hipErrorInvalidValue;
}

void launch_decode_extract_rec_gated(
    int nv,
    int khd,
    int vhd,
    float eps,
    const float* rec_out,
    const float* z,
    const float* norm_w,
    float* rec_state,
    float* attn_out,
    hipStream_t stream,
    int compact = 0) {
    hipLaunchKernelGGL(
        supersonic_qwen35_decode_extract_rec_gated_kernel,
        dim3(static_cast<unsigned int>(nv > 0 ? nv : 1)),
        dim3(256),
        0,
        stream,
        nv,
        khd,
        vhd,
        eps,
        rec_out,
        z,
        norm_w,
        rec_state,
        attn_out,
        compact);
}

struct DecodeFullAttnScratch {
    hip_bfloat16* q = nullptr;
    hip_bfloat16* k = nullptr;
    hip_bfloat16* v = nullptr;
    float* attn = nullptr;
    float* part_m = nullptr;
    float* part_l = nullptr;
    float* part_acc = nullptr;
    int nh = 0;
    int nkv = 0;
    int hd = 0;
    int max_t = 0;
    int n_splits = 0;
    int device_ordinal = -1;
};

DecodeFullAttnScratch& decode_full_attn_scratch() {
    static DecodeFullAttnScratch s;
    return s;
}

hipError_t ensure_decode_full_attn_scratch(
    int ordinal, int nh, int nkv, int hd, int max_t) {
    DecodeFullAttnScratch& s = decode_full_attn_scratch();
    constexpr int kSplits = 64;
    if (s.device_ordinal == ordinal && s.q != nullptr && s.nh == nh &&
        s.nkv == nkv && s.hd == hd &&
        s.max_t >= max_t && s.n_splits == kSplits) {
        return hipSuccess;
    }
    if (s.q != nullptr) {
        if (s.device_ordinal < 0) {
            supersonic_gpu_integrity_fail_stop(
                "decode attention scratch missing owner",
                static_cast<int>(hipErrorInvalidDevice),
                -1);
        }
        const int old_device = s.device_ordinal;
        ScopedHipDevice old_owner(old_device);
        if (!old_owner.ok()) {
            supersonic_gpu_integrity_fail_stop(
                "decode attention scratch owner switch",
                static_cast<int>(old_owner.status),
                old_device);
        }
        hipError_t err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "decode attention scratch synchronize", static_cast<int>(err), old_device);
        }
        err = hipFree(s.q);
        if (err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "decode attention scratch free", static_cast<int>(err), old_device);
        }
        s = DecodeFullAttnScratch{};
        err = old_owner.restore();
        if (err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "decode attention scratch owner restore", static_cast<int>(err), old_device);
        }
    }
    ScopedHipDevice target(ordinal);
    if (!target.ok()) {
        return target.status;
    }
    const size_t q_n = static_cast<size_t>(nh) * hd;
    const size_t slots = static_cast<size_t>(nh) * kSplits;
    const size_t part_acc_n = slots * 32 * 8;
    const size_t bytes =
        q_n * sizeof(hip_bfloat16) + q_n * sizeof(float) +
        (slots + slots + part_acc_n) * sizeof(float);
    void* base = nullptr;
    hipError_t err = hipMalloc(&base, bytes);
    if (err != hipSuccess) {
        const hipError_t restore_err = target.restore();
        if (restore_err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "decode attention scratch target restore",
                static_cast<int>(restore_err),
                ordinal);
        }
        return err;
    }
    auto* bf = static_cast<hip_bfloat16*>(base);
    s.q = bf;
    s.attn = reinterpret_cast<float*>(s.q + q_n);
    s.part_m = s.attn + q_n;
    s.part_l = s.part_m + slots;
    s.part_acc = s.part_l + slots;
    s.k = nullptr;
    s.v = nullptr;
    s.nh = nh;
    s.nkv = nkv;
    s.hd = hd;
    s.max_t = max_t;
    s.n_splits = kSplits;
    s.device_ordinal = ordinal;
    const hipError_t restore_err = target.restore();
    if (restore_err != hipSuccess) {
        supersonic_gpu_integrity_fail_stop(
            "decode attention scratch target restore", static_cast<int>(restore_err), ordinal);
    }
    return hipSuccess;
}

hipError_t launch_host_full_attn(
    int ordinal,
    const GqhMixerLayer& mx,
    float* ws_proj,
    float* saved_gate,
    int seq_off,
    hipStream_t stream,
    int batch_size = 1,
    int proj_buf_floats = 0,
    int attn_scratch_floats = 0) {
    const int nh = mx.attn_heads;
    const int nkv = mx.attn_kv_heads;
    const int hd = mx.attn_head_dim;
    const int max_t = mx.kv_max_t;
    if (batch_size <= 0) {
        batch_size = 1;
    }
    const int kv_len0 = seq_off + batch_size;
    if (nh <= 0 || nkv <= 0 || hd <= 0 || max_t <= 0 ||
        mx.kv_cache_k == nullptr || mx.kv_cache_v == nullptr ||
        seq_off < 0 || kv_len0 <= 0 || kv_len0 > max_t) {
        return hipErrorInvalidValue;
    }
    hipError_t err = ensure_decode_full_attn_scratch(ordinal, nh, nkv, hd, max_t);
    if (err != hipSuccess) {
        return err;
    }
    DecodeFullAttnScratch& sc = decode_full_attn_scratch();
    const int q_n = nh * hd;
    const int bs = 256;
    const int64_t proj_stride =
        proj_buf_floats > 0 ? static_cast<int64_t>(proj_buf_floats) : 0;
    const int64_t gate_stride =
        attn_scratch_floats > 0 ? static_cast<int64_t>(attn_scratch_floats) : 0;
    for (int b = 0; b < batch_size; ++b) {
        float* proj_b = ws_proj + static_cast<int64_t>(b) * proj_stride;
        float* gate_b = saved_gate + static_cast<int64_t>(b) * gate_stride;
        // Negative kv_len: kernel reads qwen35_split_seqlen + (-kv_len)
        // so HIP graph replay can pick up the current position.
        const int kv_len = -(b + 1);
        hipLaunchKernelGGL(
            supersonic_qwen35_pack_full_q_interleaved_kernel,
            dim3(static_cast<unsigned int>((q_n + bs - 1) / bs)),
            dim3(bs),
            0,
            stream,
            proj_b,
            sc.q,
            nh,
            hd);
        const int groups = nh / nkv;
        const float scale = 1.0f / sqrtf(static_cast<float>(hd));
        const int n_splits = sc.n_splits > 0 ? sc.n_splits : 32;
        if (groups == 6 && hd == 256 && (nh % 3) == 0) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    (supersonic_qwen35_full_attention_decode_split_gqa_kernel<
                        hip_bfloat16,
                        3>)),
                dim3(static_cast<unsigned int>(nh / 3),
                     static_cast<unsigned int>(n_splits)),
                dim3(32),
                0,
                stream,
                kv_len,
                max_t,
                hd,
                groups,
                n_splits,
                scale,
                sc.q,
                static_cast<const hip_bfloat16*>(mx.kv_cache_k),
                static_cast<const hip_bfloat16*>(mx.kv_cache_v),
                sc.part_m,
                sc.part_l,
                sc.part_acc);
        } else {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(
                    supersonic_qwen35_full_attention_decode_split_kernel<hip_bfloat16>),
                dim3(static_cast<unsigned int>(nh), static_cast<unsigned int>(n_splits)),
                dim3(32),
                0,
                stream,
                kv_len,
                max_t,
                hd,
                groups,
                n_splits,
                scale,
                sc.q,
                static_cast<const hip_bfloat16*>(mx.kv_cache_k),
                static_cast<const hip_bfloat16*>(mx.kv_cache_v),
                sc.part_m,
                sc.part_l,
                sc.part_acc);
        }
        err = hipGetLastError();
        if (err != hipSuccess) {
            return err;
        }
        hipLaunchKernelGGL(
            supersonic_qwen35_full_attention_decode_merge_kernel,
            dim3(static_cast<unsigned int>(nh)),
            dim3(32),
            0,
            stream,
            n_splits,
            hd,
            sc.part_m,
            sc.part_l,
            sc.part_acc,
            sc.attn);
        err = hipGetLastError();
        if (err != hipSuccess) {
            return err;
        }
        hipLaunchKernelGGL(
            supersonic_qwen35_full_attn_gate_f32_kernel,
            dim3(static_cast<unsigned int>((q_n + bs - 1) / bs)),
            dim3(bs),
            0,
            stream,
            sc.attn,
            gate_b,
            proj_b,
            q_n);
    }
    return hipGetLastError();
}

void launch_swiglu_f32(
    float* gate,
    const float* up,
    int n,
    hipStream_t stream,
    int ncols = 1,
    int64_t col_stride = 0) {
    if (ncols <= 0) {
        ncols = 1;
    }
    const int64_t stride = col_stride > 0 ? col_stride : static_cast<int64_t>(n);
    const int bs = 256;
    const int gs = (n + bs - 1) / bs;
    for (int c = 0; c < ncols; ++c) {
        hipLaunchKernelGGL(
            supersonic_qwen35_swiglu_f32_kernel,
            dim3(static_cast<unsigned int>(gs)),
            dim3(bs),
            0,
            stream,
            gate + static_cast<int64_t>(c) * stride,
            up + static_cast<int64_t>(c) * stride,
            n);
    }
}

hipError_t ensure_decode_rms_partials(int ordinal, float** out) {
    if (out == nullptr || ordinal < 0) {
        return hipErrorInvalidValue;
    }
    static float* p = nullptr;
    static int owner = -1;
    if (p != nullptr && owner != ordinal) {
        if (owner < 0) {
            supersonic_gpu_integrity_fail_stop(
                "decode RMS scratch missing owner", static_cast<int>(hipErrorInvalidDevice), -1);
        }
        const int old_device = owner;
        ScopedHipDevice old_owner(old_device);
        if (!old_owner.ok()) {
            supersonic_gpu_integrity_fail_stop(
                "decode RMS scratch owner switch", static_cast<int>(old_owner.status), old_device);
        }
        hipError_t err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "decode RMS scratch synchronize", static_cast<int>(err), old_device);
        }
        err = hipFree(p);
        if (err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "decode RMS scratch free", static_cast<int>(err), old_device);
        }
        p = nullptr;
        owner = -1;
        err = old_owner.restore();
        if (err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "decode RMS scratch owner restore", static_cast<int>(err), old_device);
        }
    }
    if (p == nullptr) {
        ScopedHipDevice target(ordinal);
        if (!target.ok()) {
            return target.status;
        }
        const hipError_t err = hipMalloc(&p, 256 * sizeof(float));
        if (err != hipSuccess) {
            p = nullptr;
            const hipError_t restore_err = target.restore();
            if (restore_err != hipSuccess) {
                supersonic_gpu_integrity_fail_stop(
                    "decode RMS scratch target restore", static_cast<int>(restore_err), ordinal);
            }
            return err;
        }
        owner = ordinal;
        const hipError_t restore_err = target.restore();
        if (restore_err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "decode RMS scratch target restore", static_cast<int>(restore_err), ordinal);
        }
    }
    *out = p;
    return hipSuccess;
}

hipError_t launch_decode_rms(
    float* hidden_f32,
    float* normed,
    const hip_bfloat16* hidden_io,
    const hip_bfloat16* norm_w,
    int hidden_dim,
    int batch_size,
    float eps,
    float unit_offset,
    int flags,
    int ordinal,
    hipStream_t stream) {
    constexpr int kBs = 256;
    float* partials = nullptr;
    const hipError_t partials_err = ensure_decode_rms_partials(ordinal, &partials);
    if (partials_err != hipSuccess) {
        return partials_err;
    }
    if (partials != nullptr && hidden_dim == 5120) {
        const int npartials = (hidden_dim + kBs - 1) / kBs;
        const int B = batch_size > 0 ? batch_size : 1;
        const dim3 pgrid(
            static_cast<unsigned int>(npartials),
            static_cast<unsigned int>(B));
        hipLaunchKernelGGL(
            supersonic_qwen35_decode_rms_partial_kernel,
            pgrid,
            dim3(kBs),
            kBs * sizeof(float),
            stream,
            hidden_f32,
            hidden_io,
            partials,
            hidden_dim,
            eps,
            flags);
        hipLaunchKernelGGL(
            supersonic_qwen35_decode_rms_scale_kernel,
            pgrid,
            dim3(kBs),
            0,
            stream,
            hidden_f32,
            normed,
            norm_w,
            partials,
            hidden_dim,
            npartials,
            eps,
            unit_offset);
        return hipGetLastError();
    }
    const unsigned int rms_grid =
        static_cast<unsigned int>(batch_size > 1 ? batch_size : 1);
    hipLaunchKernelGGL(
        supersonic_qwen35_decode_rms_kernel,
        dim3(rms_grid),
        dim3(kBs),
        kBs * sizeof(float),
        stream,
        hidden_f32,
        normed,
        hidden_io,
        norm_w,
        hidden_dim,
        batch_size,
        eps,
        unit_offset,
        flags);
    return hipGetLastError();
}

hipError_t launch_hidden_store_bf16(
    const float* hidden_f32,
    hip_bfloat16* hidden_io,
    int n,
    hipStream_t stream) {
    const int bs = 256;
    const int gs = (n + bs - 1) / bs;
    hipLaunchKernelGGL(
        supersonic_qwen35_pack_f32_to_bf16_kernel,
        dim3(static_cast<unsigned int>(gs)),
        dim3(bs),
        0,
        stream,
        hidden_f32,
        hidden_io,
        n);
    return hipGetLastError();
}

namespace {
hip_bfloat16* g_ggml_x_bf = nullptr;
hip_bfloat16* g_ggml_y_bf = nullptr;
int g_ggml_cap_in = 0;
int g_ggml_cap_out = 0;
int g_ggml_device_ordinal = -1;
}  // namespace

hipError_t ensure_ggml_k_gemv_scratch(int ordinal, int in_dim, int out_dim) {
    if (g_ggml_device_ordinal < 0 &&
        (g_ggml_x_bf != nullptr || g_ggml_y_bf != nullptr)) {
        supersonic_gpu_integrity_fail_stop(
            "GGML-K scratch missing owner", static_cast<int>(hipErrorInvalidDevice), -1);
    }
    if (g_ggml_device_ordinal >= 0 && g_ggml_device_ordinal != ordinal) {
        const int old_device = g_ggml_device_ordinal;
        ScopedHipDevice old_owner(old_device);
        if (!old_owner.ok()) {
            supersonic_gpu_integrity_fail_stop(
                "GGML-K scratch owner switch", static_cast<int>(old_owner.status), old_device);
        }
        hipError_t err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "GGML-K scratch synchronize", static_cast<int>(err), old_device);
        }
        if (g_ggml_x_bf != nullptr) {
            err = hipFree(g_ggml_x_bf);
            if (err != hipSuccess) {
                supersonic_gpu_integrity_fail_stop(
                    "GGML-K input scratch free", static_cast<int>(err), old_device);
            }
            g_ggml_x_bf = nullptr;
            g_ggml_cap_in = 0;
        }
        if (g_ggml_y_bf != nullptr) {
            err = hipFree(g_ggml_y_bf);
            if (err != hipSuccess) {
                supersonic_gpu_integrity_fail_stop(
                    "GGML-K output scratch free", static_cast<int>(err), old_device);
            }
            g_ggml_y_bf = nullptr;
            g_ggml_cap_out = 0;
        }
        g_ggml_device_ordinal = -1;
        err = old_owner.restore();
        if (err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "GGML-K scratch owner restore", static_cast<int>(err), old_device);
        }
    }

    ScopedHipDevice target(ordinal);
    if (!target.ok()) {
        return target.status;
    }
    auto finish = [&](hipError_t err) {
        const hipError_t restore_err = target.restore();
        if (restore_err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "GGML-K scratch target restore", static_cast<int>(restore_err), ordinal);
        }
        return err;
    };
    g_ggml_device_ordinal = ordinal;
    if (in_dim > g_ggml_cap_in) {
        if (g_ggml_x_bf) {
            hipError_t err = hipDeviceSynchronize();
            if (err != hipSuccess) {
                supersonic_gpu_integrity_fail_stop(
                    "GGML-K input scratch synchronize", static_cast<int>(err), ordinal);
            }
            err = hipFree(g_ggml_x_bf);
            if (err != hipSuccess) {
                supersonic_gpu_integrity_fail_stop(
                    "GGML-K input scratch free", static_cast<int>(err), ordinal);
            }
            g_ggml_x_bf = nullptr;
            g_ggml_cap_in = 0;
        }
        const hipError_t err = hipMalloc(
            &g_ggml_x_bf,
            static_cast<size_t>(in_dim) * sizeof(hip_bfloat16));
        if (err != hipSuccess) {
            return finish(err);
        }
        g_ggml_cap_in = in_dim;
    }
    if (out_dim > g_ggml_cap_out) {
        if (g_ggml_y_bf) {
            hipError_t err = hipDeviceSynchronize();
            if (err != hipSuccess) {
                supersonic_gpu_integrity_fail_stop(
                    "GGML-K output scratch synchronize", static_cast<int>(err), ordinal);
            }
            err = hipFree(g_ggml_y_bf);
            if (err != hipSuccess) {
                supersonic_gpu_integrity_fail_stop(
                    "GGML-K output scratch free", static_cast<int>(err), ordinal);
            }
            g_ggml_y_bf = nullptr;
            g_ggml_cap_out = 0;
        }
        const hipError_t err = hipMalloc(
            &g_ggml_y_bf,
            static_cast<size_t>(out_dim) * sizeof(hip_bfloat16));
        if (err != hipSuccess) {
            return finish(err);
        }
        g_ggml_cap_out = out_dim;
    }
    return finish(hipSuccess);
}

template <bool kAcc>
hipError_t launch_ggml_k_gemv_impl(
    const GqhProjHdr& h,
    const float* x,
    float* y,
    int in_dim,
    int out_dim,
    hipStream_t stream,
    int ncols = 1,
    int64_t x_col_stride = 0,
    int64_t y_col_stride = 0) {
    if (out_dim <= 0 || in_dim <= 0 || h.wire == nullptr || x == nullptr ||
        y == nullptr) {
        return hipErrorInvalidValue;
    }
    if (h.qtype == 8 && (in_dim % 32) != 0) {
        return hipErrorInvalidValue;
    }
    if (h.qtype != 8 && (in_dim % 256) != 0) {
        return hipErrorInvalidValue;
    }
    if (ncols <= 0) {
        ncols = 1;
    }
    if (x_col_stride <= 0) {
        x_col_stride = in_dim;
    }
    if (y_col_stride <= 0) {
        y_col_stride = out_dim;
    }
    // Same fused F32 walker as the residual-add path. Dual-row when the
    // output is a multiple of 8 so both rows stay in-bounds.
    constexpr int kWarps = 4;
    constexpr int kThreads = kWarps * 32;
    const bool dual = (out_dim % (kWarps * 2)) == 0 && out_dim >= 4096;
    const int rows_per_block = dual ? kWarps * 2 : kWarps;
    const dim3 blocks(static_cast<unsigned int>(
        (out_dim + rows_per_block - 1) / rows_per_block));
    const dim3 threads(kThreads);
    auto* packed = static_cast<const uint8_t*>(h.wire);
    const bool skinny = ncols > 1 && ncols <= 4;
    const int kC = skinny ? (ncols <= 3 ? 3 : 4) : 1;
    auto launch = [&](auto kernel) {
        hipLaunchKernelGGL(
            kernel, blocks, threads, 0, stream, packed, x, y, in_dim, out_dim,
            ncols, x_col_stride, y_col_stride);
    };
    switch (h.qtype) {
        case 8:
            if (skinny && dual && kC == 3) {
                launch(HIP_KERNEL_NAME(
                    (supersonic_qwen35_ggml_k_matvec_kernel<8, kAcc, 2, 3>)));
            } else if (skinny && dual) {
                launch(HIP_KERNEL_NAME(
                    (supersonic_qwen35_ggml_k_matvec_kernel<8, kAcc, 2, 4>)));
            } else if (skinny && kC == 3) {
                launch(HIP_KERNEL_NAME(
                    (supersonic_qwen35_ggml_k_matvec_kernel<8, kAcc, 1, 3>)));
            } else if (skinny) {
                launch(HIP_KERNEL_NAME(
                    (supersonic_qwen35_ggml_k_matvec_kernel<8, kAcc, 1, 4>)));
            } else if (dual) {
                launch(HIP_KERNEL_NAME(
                    (supersonic_qwen35_ggml_k_matvec_kernel<8, kAcc, 2, 1>)));
            } else {
                launch(HIP_KERNEL_NAME(
                    (supersonic_qwen35_ggml_k_matvec_kernel<8, kAcc, 1, 1>)));
            }
            break;
        case 12:
            if (skinny && dual) {
                launch(HIP_KERNEL_NAME(
                    (supersonic_qwen35_ggml_k_matvec_kernel<12, kAcc, 2, 4>)));
            } else if (skinny) {
                launch(HIP_KERNEL_NAME(
                    (supersonic_qwen35_ggml_k_matvec_kernel<12, kAcc, 1, 4>)));
            } else if (dual) {
                launch(HIP_KERNEL_NAME(
                    (supersonic_qwen35_ggml_k_matvec_kernel<12, kAcc, 2, 1>)));
            } else {
                launch(HIP_KERNEL_NAME(
                    (supersonic_qwen35_ggml_k_matvec_kernel<12, kAcc, 1, 1>)));
            }
            break;
        case 13:
            if (skinny && dual) {
                launch(HIP_KERNEL_NAME(
                    (supersonic_qwen35_ggml_k_matvec_kernel<13, kAcc, 2, 4>)));
            } else if (skinny) {
                launch(HIP_KERNEL_NAME(
                    (supersonic_qwen35_ggml_k_matvec_kernel<13, kAcc, 1, 4>)));
            } else if (dual) {
                launch(HIP_KERNEL_NAME(
                    (supersonic_qwen35_ggml_k_matvec_kernel<13, kAcc, 2, 1>)));
            } else {
                launch(HIP_KERNEL_NAME(
                    (supersonic_qwen35_ggml_k_matvec_kernel<13, kAcc, 1, 1>)));
            }
            break;
        case 14:
            if (skinny && dual) {
                launch(HIP_KERNEL_NAME(
                    (supersonic_qwen35_ggml_k_matvec_kernel<14, kAcc, 2, 4>)));
            } else if (skinny) {
                launch(HIP_KERNEL_NAME(
                    (supersonic_qwen35_ggml_k_matvec_kernel<14, kAcc, 1, 4>)));
            } else if (dual) {
                launch(HIP_KERNEL_NAME(
                    (supersonic_qwen35_ggml_k_matvec_kernel<14, kAcc, 2, 1>)));
            } else {
                launch(HIP_KERNEL_NAME(
                    (supersonic_qwen35_ggml_k_matvec_kernel<14, kAcc, 1, 1>)));
            }
            break;
        default:
            return hipErrorInvalidValue;
    }
    return hipGetLastError();
}

hipError_t launch_ggml_k_gemv(
    const GqhProjHdr& h,
    const float* x,
    float* y,
    int in_dim,
    int out_dim,
    hipStream_t stream,
    int ncols = 1,
    int64_t x_col_stride = 0,
    int64_t y_col_stride = 0) {
    return launch_ggml_k_gemv_impl<false>(
        h, x, y, in_dim, out_dim, stream, ncols, x_col_stride, y_col_stride);
}

hipError_t launch_ggml_k_gemv_acc(
    const GqhProjHdr& h,
    const float* x,
    float* y,
    int in_dim,
    int out_dim,
    hipStream_t stream,
    int ncols = 1,
    int64_t x_col_stride = 0,
    int64_t y_col_stride = 0) {
    return launch_ggml_k_gemv_impl<true>(
        h, x, y, in_dim, out_dim, stream, ncols, x_col_stride, y_col_stride);
}

hipError_t launch_mixer_proj_acc(
    int ordinal,
    const GqhProjHdr& h,
    const float* x,
    float* y,
    int in_dim,
    int out_dim,
    hipStream_t stream,
    int ncols = 1,
    int64_t x_col_stride = 0,
    int64_t y_col_stride = 0) {
    if (out_dim <= 0 || h.wire == nullptr) {
        return hipSuccess;
    }
    if (ncols <= 0) {
        ncols = 1;
    }
    if (x_col_stride <= 0) {
        x_col_stride = in_dim;
    }
    if (y_col_stride <= 0) {
        y_col_stride = out_dim;
    }
    if (h.rung >= 0) {
        return launch_gqh_gemv_acc(
            ordinal, h, x, y, in_dim, out_dim, stream, ncols, x_col_stride,
            y_col_stride);
    }
    if (ggml_k_qtype(h.qtype)) {
        return launch_ggml_k_gemv_acc(
            h, x, y, in_dim, out_dim, stream, ncols, x_col_stride,
            y_col_stride);
    }
    if (mix_qtype(h.qtype)) {
        for (int c = 0; c < ncols; ++c) {
            const int st = supersonic_gqh_hip_mix_matvec_stream(
                ordinal,
                h.qtype,
                h.wire,
                x + static_cast<int64_t>(c) * x_col_stride,
                y + static_cast<int64_t>(c) * y_col_stride,
                in_dim,
                out_dim,
                1,
                1,
                h.mix_mode,
                h.mix_lut,
                stream);
            if (st != 0) {
                return hipErrorInvalidValue;
            }
        }
        return hipSuccess;
    }
    return hipSuccess;
}

hipError_t launch_mixer_proj(
    int ordinal,
    const GqhProjHdr& h,
    const float* x,
    float* y,
    int in_dim,
    int out_dim,
    hipStream_t stream,
    bool do_round = true,
    int ncols = 1,
    int64_t x_col_stride = 0,
    int64_t y_col_stride = 0) {
    if (out_dim <= 0 || h.wire == nullptr) {
        return hipSuccess;
    }
    if (ncols <= 0) {
        ncols = 1;
    }
    if (x_col_stride <= 0) {
        x_col_stride = in_dim;
    }
    if (y_col_stride <= 0) {
        y_col_stride = out_dim;
    }
    hipError_t err = hipSuccess;
    if (h.rung >= 0) {
        err = launch_gqh_gemv(
            ordinal, h, x, y, in_dim, out_dim, stream, ncols, x_col_stride,
            y_col_stride);
    } else if (ggml_k_qtype(h.qtype)) {
        err = launch_ggml_k_gemv(
            h, x, y, in_dim, out_dim, stream, ncols, x_col_stride,
            y_col_stride);
    } else if (mix_qtype(h.qtype)) {
        for (int c = 0; c < ncols; ++c) {
            const int st = supersonic_gqh_hip_mix_matvec_stream(
                ordinal,
                h.qtype,
                h.wire,
                x + static_cast<int64_t>(c) * x_col_stride,
                y + static_cast<int64_t>(c) * y_col_stride,
                in_dim,
                out_dim,
                1,
                0,
                h.mix_mode,
                h.mix_lut,
                stream);
            if (st != 0) {
                return hipErrorInvalidValue;
            }
        }
    } else {
        return hipSuccess;
    }
    if (err != hipSuccess) {
        return err;
    }
    if (do_round) {
        launch_round_f32(y, out_dim, stream, ncols, y_col_stride);
    }
    return hipSuccess;
}

}  // namespace

namespace {

int prefill_backend_failure(int project_status, hipError_t native_status) {
    return static_cast<int>(
        0x80000000u
        | ((static_cast<uint32_t>(project_status) & 0x7fffu) << 16)
        | (static_cast<uint32_t>(native_status) & 0xffffu));
}

hipError_t clear_split_graph_cache(SplitGraphCache& cache, bool destroy_stream) {
    const int owner = cache.device_ordinal;
    if (owner >= 0) {
        ScopedHipDevice scoped(owner);
        if (!scoped.ok()) {
            supersonic_gpu_integrity_fail_stop(
                "split graph owner switch", static_cast<int>(scoped.status), owner);
        }
        if (cache.stream != nullptr) {
            const hipError_t err = hipStreamSynchronize(cache.stream);
            if (err != hipSuccess) {
                supersonic_gpu_integrity_fail_stop(
                    "split graph stream synchronize", static_cast<int>(err), owner);
            }
        }
        // Graph replay and the graph stream are process-global. Synchronize
        // the owning device before destroying either object, including when
        // invalidation is called from another device context.
        const hipError_t sync_err = hipDeviceSynchronize();
        if (sync_err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "split graph device synchronize", static_cast<int>(sync_err), owner);
        }
        if (cache.exec != nullptr) {
            const hipError_t err = hipGraphExecDestroy(cache.exec);
            if (err != hipSuccess) {
                supersonic_gpu_integrity_fail_stop(
                    "split graph exec destroy", static_cast<int>(err), owner);
            }
            cache.exec = nullptr;
        }
        if (cache.graph != nullptr) {
            const hipError_t err = hipGraphDestroy(cache.graph);
            if (err != hipSuccess) {
                supersonic_gpu_integrity_fail_stop(
                    "split graph destroy", static_cast<int>(err), owner);
            }
            cache.graph = nullptr;
        }
        if (destroy_stream && cache.stream != nullptr) {
            const hipError_t err = hipStreamDestroy(cache.stream);
            if (err != hipSuccess) {
                supersonic_gpu_integrity_fail_stop(
                    "split graph stream destroy", static_cast<int>(err), owner);
            }
            cache.stream = nullptr;
        }
        // Do not report success until the caller's current device has been
        // restored. A restore failure is unrecoverable while graph metadata
        // still names the owner and is handled by the integrity policy.
        const hipError_t restore_err = scoped.restore();
        if (restore_err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "split graph owner restore", static_cast<int>(restore_err), owner);
        }
    } else {
        if (cache.exec != nullptr || cache.graph != nullptr || cache.stream != nullptr) {
            supersonic_gpu_integrity_fail_stop(
                "split graph resources missing owner",
                static_cast<int>(hipErrorInvalidDevice),
                owner);
        }
    }
    if (destroy_stream || cache.stream == nullptr) {
        cache.device_ordinal = -1;
    }
    cache.num_layers = -1;
    cache.grid_in = cache.grid_mid = cache.grid_out = 0;
    cache.grid_gate = cache.grid_up = cache.grid_down = 0;
    cache.layers = nullptr;
    cache.hidden_io = nullptr;
    cache.workspace = nullptr;
    cache.counters = nullptr;
    cache.barrier_counter = nullptr;
    cache.barrier_flag = nullptr;
    cache.int4 = nullptr;
    cache.cos_table = nullptr;
    cache.sin_table = nullptr;
    cache.fp8_scales = nullptr;
    cache.kv_fp8_descs = nullptr;
    cache.batch_descs = nullptr;
    cache.state_signature = 0;
    cache.batch_size = 0;
    return hipSuccess;
}

int linear_prefill_block_override() {
    const char* value = std::getenv("DOTCACHE_QWEN38_HIP_FUSED_PREFILL_BLOCK");
    if (value == nullptr || *value == '\0') {
        return 0;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || parsed <= 0) {
        return 0;
    }
    if (parsed < 32) {
        return 32;
    }
    if (parsed > 512) {
        return 512;
    }
    return static_cast<int>(parsed);
}

hipError_t maybe_sync() {
    const char* value = std::getenv("SUPERSONIC_SYNC_EACH_KERNEL");
    const bool enabled = value != nullptr && value[0] != '\0' && value[0] != '0';
    return enabled ? hipDeviceSynchronize() : hipSuccess;
}

template <typename T>
int full_attention_prefill_device(
    int device_ordinal,
    int batch_size,
    int q_heads,
    int kv_heads,
    int q_len,
    int kv_len,
    int head_dim,
    int num_kv_groups,
    float scale,
    int seqlen_offset,
    const void* query,
    const void* key,
    const void* value,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);

    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_ordinal) != hipSuccess) {
        return 1;
    }

    const T* d_query = static_cast<const T*>(query);
    const T* d_key = static_cast<const T*>(key);
    const T* d_value = static_cast<const T*>(value);
    float* d_out = static_cast<float*>(out);
    static unsigned int* d_row_counter = nullptr;
    if (d_row_counter == nullptr) {
        if (hipMalloc(&d_row_counter, sizeof(unsigned int)) != hipSuccess) return 2;
    }
    if (hipMemset(d_row_counter, 0, sizeof(unsigned int)) != hipSuccess) return 10;

    // RDNA3 `multiProcessorCount` reports WGPs, not CUs. The kernel is
    // one warp per query row via an atomic counter; at 8k×24 heads we
    // need thousands of warps in flight, not 1–2 per CU.
    int grid = props.multiProcessorCount > 0 ? props.multiProcessorCount : 1;
    {
        const char* arch = props.gcnArchName;
        const bool is_rdna3_wgp_arch =
            arch[0] == 'g' && arch[1] == 'f' && arch[2] == 'x' &&
            arch[3] == '1' && arch[4] == '1';
        if (is_rdna3_wgp_arch) grid *= 2;
    }
    const int total_rows = batch_size * q_heads * q_len;
    grid = std::min(total_rows, std::max(grid * 32, 1024));
    const int block = props.warpSize > 0 ? props.warpSize : 32;
    if (head_dim > block * 8) return 14;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_full_attention_prefill_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_size,
        q_heads,
        kv_heads,
        q_len,
        kv_len,
        head_dim,
        num_kv_groups,
        scale,
        seqlen_offset,
        d_query,
        d_key,
        d_value,
        d_out,
        d_row_counter);
    if (hipGetLastError() != hipSuccess) return 11;
    if (maybe_sync() != hipSuccess) return 12;
    return 0;
}

template <typename T>
int linear_prefill_conv_pack_device(
    int device_ordinal,
    int batch_size,
    int conv_dim,
    int total_len,
    int seq_len,
    int kernel_size,
    const void* mixed_qkv,
    const void* weights,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const size_t out_elems = static_cast<size_t>(batch_size) * static_cast<size_t>(seq_len) *
        static_cast<size_t>(conv_dim);
    const unsigned int grid = static_cast<unsigned int>((out_elems + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_linear_prefill_conv_pack_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_size,
        conv_dim,
        total_len,
        seq_len,
        kernel_size,
        static_cast<const T*>(mixed_qkv),
        static_cast<const T*>(weights),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 60;
    if (maybe_sync() != hipSuccess) return 61;
    return 0;
}

template <typename T>
int delta_recurrent_prefill_device(
    int device_ordinal,
    int batch_heads,
    int seq_len,
    int k_head_dim,
    int v_head_dim,
    const void* initial_state,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256) return 69;
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_delta_recurrent_prefill_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        seq_len,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(initial_state),
        static_cast<const T*>(query),
        static_cast<const T*>(key),
        static_cast<const T*>(value),
        static_cast<const T*>(beta),
        static_cast<const T*>(g),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 67;
    if (maybe_sync() != hipSuccess) return 68;
    return 0;
}

template <typename T>
int delta_chunk_single_prefill_device(
    int device_ordinal,
    int batch_heads,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (chunk_size > 64 || k_head_dim > 256) return 76;
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_delta_chunk_single_prefill_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        chunk_size,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(query),
        static_cast<const T*>(key),
        static_cast<const T*>(value),
        static_cast<const T*>(beta),
        static_cast<const T*>(g),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 77;
    if (maybe_sync() != hipSuccess) return 78;
    return 0;
}

template <typename T>
int delta_chunk_step_device(
    int device_ordinal,
    int batch_heads,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const void* prev_state,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256) return 80;
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_delta_chunk_step_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        chunk_size,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(prev_state),
        static_cast<const T*>(query),
        static_cast<const T*>(key),
        static_cast<const T*>(value),
        static_cast<const T*>(beta),
        static_cast<const T*>(g),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 81;
    if (maybe_sync() != hipSuccess) return 82;
    return 0;
}

template <typename T>
int delta_chunk_scan_raw_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const void* initial_state,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256) return 83;
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_delta_chunk_scan_raw_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(initial_state),
        static_cast<const T*>(query),
        static_cast<const T*>(key),
        static_cast<const T*>(value),
        static_cast<const T*>(beta),
        static_cast<const T*>(g),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 84;
    if (maybe_sync() != hipSuccess) return 85;
    return 0;
}

template <typename T>
int delta_state_scan_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const void* initial_state,
    const void* packed_scan,
    const void* value,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256) return 88;
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_delta_state_scan_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(initial_state),
        static_cast<const T*>(packed_scan),
        static_cast<const T*>(value),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 89;
    if (maybe_sync() != hipSuccess) return 96;
    return 0;
}

template <typename T>
int delta_chunk_fused_device(
    int device_ordinal,
    int batch_heads,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const void* prev_state,
    const void* packed_chunk,
    const void* value,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256 || chunk_size > 64) return 97;
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_delta_chunk_fused_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        chunk_size,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(prev_state),
        static_cast<const T*>(packed_chunk),
        static_cast<const T*>(value),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 98;
    if (maybe_sync() != hipSuccess) return 99;
    return 0;
}

template <typename T>
int delta_full_scan_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const void* initial_state,
    const void* weighted_key_scan,
    const void* k_cumdecay_scan,
    const void* q_state_scan,
    const void* local_attn_scan,
    const void* state_decay_scan,
    const void* value,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256 || chunk_size > 64) return 100;
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_delta_full_scan_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(initial_state),
        static_cast<const T*>(weighted_key_scan),
        static_cast<const T*>(k_cumdecay_scan),
        static_cast<const T*>(q_state_scan),
        static_cast<const T*>(local_attn_scan),
        static_cast<const T*>(state_decay_scan),
        static_cast<const T*>(value),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 101;
    if (maybe_sync() != hipSuccess) return 102;
    return 0;
}

template <typename T>
int delta_local_attn_scan_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    const void* query_scan,
    const void* key_scan,
    const void* exp_g_scan,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256 || chunk_size > 64) return 112;
    if (chunk_size <= 4) {
        constexpr int block = 256;
        const size_t total =
            static_cast<size_t>(batch_heads) * static_cast<size_t>(num_chunks) *
            static_cast<size_t>(chunk_size) * static_cast<size_t>(chunk_size);
        const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(supersonic_qwen35_delta_local_attn_scan_flat_kernel<T>),
            dim3(grid),
            dim3(block),
            0,
            0,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            static_cast<const T*>(query_scan),
            static_cast<const T*>(key_scan),
            static_cast<const T*>(exp_g_scan),
            static_cast<T*>(out));
    } else {
        const unsigned int block = chunk_size <= 32 ? 32u : 64u;
        const size_t total_rows =
            static_cast<size_t>(batch_heads) * static_cast<size_t>(num_chunks) *
            static_cast<size_t>(chunk_size);
        const unsigned int grid = static_cast<unsigned int>(total_rows);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(supersonic_qwen35_delta_local_attn_scan_row_kernel<T>),
            dim3(grid),
            dim3(block),
            0,
            0,
            batch_heads,
            num_chunks,
            chunk_size,
            k_head_dim,
            static_cast<const T*>(query_scan),
            static_cast<const T*>(key_scan),
            static_cast<const T*>(exp_g_scan),
            static_cast<T*>(out));
    }
    if (hipGetLastError() != hipSuccess) return 113;
    if (maybe_sync() != hipSuccess) return 114;
    return 0;
}

template <typename T>
int delta_base_attn_scan_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    const void* k_beta_scan,
    const void* key_scan,
    const void* exp_g_scan,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256 || chunk_size > 64) return 115;
    constexpr int block = 256;
    const size_t total =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(num_chunks) *
        static_cast<size_t>(chunk_size) * static_cast<size_t>(chunk_size);
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_delta_base_attn_scan_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        static_cast<const T*>(k_beta_scan),
        static_cast<const T*>(key_scan),
        static_cast<const T*>(exp_g_scan),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 116;
    if (maybe_sync() != hipSuccess) return 117;
    return 0;
}

template <typename T>
int delta_attn_solve_scan_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    const void* base_attn_scan,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (chunk_size > 64) return 118;
    constexpr int block = 1;
    const unsigned int grid =
        static_cast<unsigned int>(batch_heads * num_chunks);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_delta_attn_solve_scan_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        static_cast<const T*>(base_attn_scan),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 119;
    if (maybe_sync() != hipSuccess) return 120;
    return 0;
}

template <typename T>
int delta_attn_solve_from_inputs_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    const void* k_beta_scan,
    const void* key_scan,
    const void* exp_g_scan,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (chunk_size > 64 || k_head_dim > 256) return 121;
    constexpr int block = 1;
    const unsigned int grid =
        static_cast<unsigned int>(batch_heads * num_chunks);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_delta_attn_solve_from_inputs_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        static_cast<const T*>(k_beta_scan),
        static_cast<const T*>(key_scan),
        static_cast<const T*>(exp_g_scan),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 122;
    if (maybe_sync() != hipSuccess) return 123;
    return 0;
}

template <typename T>
int swiglu_mul_device(
    int device_ordinal,
    int elem_count,
    const void* gate,
    const void* up,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid = static_cast<unsigned int>((elem_count + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_swiglu_mul_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        elem_count,
        static_cast<const T*>(gate),
        static_cast<const T*>(up),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 121;
    if (maybe_sync() != hipSuccess) return 122;
    return 0;
}

template <typename T, typename IndexT>
int embedding_lookup_device(
    int device_ordinal,
    int token_count,
    int vocab_size,
    int hidden_size,
    const void* embeddings,
    const void* indexes,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const int total_elems = token_count * hidden_size;
    const unsigned int grid = static_cast<unsigned int>((total_elems + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_embedding_lookup_kernel<T, IndexT>),
        dim3(grid),
        dim3(block),
        0,
        0,
        token_count,
        vocab_size,
        hidden_size,
        static_cast<const T*>(embeddings),
        static_cast<const IndexT*>(indexes),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 123;
    if (maybe_sync() != hipSuccess) return 124;
    return 0;
}

template <typename T>
int causal_mask_device(
    int device_ordinal,
    int batch_size,
    int tgt_len,
    int seqlen_offset,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const int kv_len = tgt_len + seqlen_offset;
    const int total_elems = batch_size * tgt_len * kv_len;
    const unsigned int grid = static_cast<unsigned int>((total_elems + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_causal_mask_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_size,
        tgt_len,
        seqlen_offset,
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 125;
    if (maybe_sync() != hipSuccess) return 126;
    return 0;
}

template <typename T>
int cumsum_last_dim_device(
    int device_ordinal,
    int rows,
    int cols,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_cumsum_last_dim_kernel<T>),
        dim3(static_cast<unsigned int>(rows)),
        dim3(1),
        0,
        0,
        rows,
        cols,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 127;
    if (maybe_sync() != hipSuccess) return 128;
    return 0;
}

template <typename T>
int exp_device(
    int device_ordinal,
    int total_elems,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<size_t>(total_elems) + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_exp_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        total_elems,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 129;
    if (maybe_sync() != hipSuccess) return 130;
    return 0;
}

template <typename T>
int recip_device(
    int device_ordinal,
    int total_elems,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<size_t>(total_elems) + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_recip_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        total_elems,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 131;
    if (maybe_sync() != hipSuccess) return 132;
    return 0;
}

template <typename T>
int sigmoid_device(
    int device_ordinal,
    int total_elems,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<size_t>(total_elems) + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_sigmoid_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        total_elems,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 133;
    if (maybe_sync() != hipSuccess) return 134;
    return 0;
}

template <typename T>
int log_device(
    int device_ordinal,
    int total_elems,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<size_t>(total_elems) + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_log_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        total_elems,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 155;
    if (maybe_sync() != hipSuccess) return 156;
    return 0;
}

template <typename In, typename Out>
int cast_device(
    int device_ordinal,
    int total_elems,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<size_t>(total_elems) + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_cast_kernel<In, Out>),
        dim3(grid),
        dim3(block),
        0,
        0,
        total_elems,
        static_cast<const In*>(xs),
        static_cast<Out*>(out));
    if (hipGetLastError() != hipSuccess) return 135;
    if (maybe_sync() != hipSuccess) return 136;
    return 0;
}

template <typename T>
int unary_view_device(
    int op,
    int device_ordinal,
    int rank,
    size_t total_elems,
    float scalar,
    const void* xs,
    const int* in_strides,
    const int* out_dims,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    int* in_strides_dev = nullptr;
    int* out_dims_dev = nullptr;
    const size_t bytes = static_cast<size_t>(rank) * sizeof(int);
    if (rank > 0) {
        if (hipMalloc(&in_strides_dev, bytes) != hipSuccess) return 158;
        if (hipMalloc(&out_dims_dev, bytes) != hipSuccess) {
            hipFree(in_strides_dev);
            return 158;
        }
        if (hipMemcpy(in_strides_dev, in_strides, bytes, hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(out_dims_dev, out_dims, bytes, hipMemcpyHostToDevice) != hipSuccess) {
            hipFree(in_strides_dev);
            hipFree(out_dims_dev);
            return 158;
        }
    }
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((total_elems + static_cast<size_t>(block) - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_unary_view_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        op,
        rank,
        total_elems,
        scalar,
        static_cast<const T*>(xs),
        in_strides_dev,
        out_dims_dev,
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) {
        if (rank > 0) {
            hipFree(in_strides_dev);
            hipFree(out_dims_dev);
        }
        return 159;
    }
    if (maybe_sync() != hipSuccess) {
        if (rank > 0) {
            hipFree(in_strides_dev);
            hipFree(out_dims_dev);
        }
        return 160;
    }
    if (rank > 0) {
        hipFree(in_strides_dev);
        hipFree(out_dims_dev);
    }
    return 0;
}

template <typename In, typename Out>
int cast_view_device(
    int device_ordinal,
    int rank,
    size_t total_elems,
    const void* xs,
    const int* in_strides,
    const int* out_dims,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    int* in_strides_dev = nullptr;
    int* out_dims_dev = nullptr;
    const size_t bytes = static_cast<size_t>(rank) * sizeof(int);
    if (rank > 0) {
        if (hipMalloc(&in_strides_dev, bytes) != hipSuccess) return 161;
        if (hipMalloc(&out_dims_dev, bytes) != hipSuccess) {
            hipFree(in_strides_dev);
            return 161;
        }
        if (hipMemcpy(in_strides_dev, in_strides, bytes, hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(out_dims_dev, out_dims, bytes, hipMemcpyHostToDevice) != hipSuccess) {
            hipFree(in_strides_dev);
            hipFree(out_dims_dev);
            return 161;
        }
    }
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((total_elems + static_cast<size_t>(block) - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_cast_view_kernel<In, Out>),
        dim3(grid),
        dim3(block),
        0,
        0,
        rank,
        total_elems,
        static_cast<const In*>(xs),
        in_strides_dev,
        out_dims_dev,
        static_cast<Out*>(out));
    if (hipGetLastError() != hipSuccess) {
        if (rank > 0) {
            hipFree(in_strides_dev);
            hipFree(out_dims_dev);
        }
        return 162;
    }
    if (maybe_sync() != hipSuccess) {
        if (rank > 0) {
            hipFree(in_strides_dev);
            hipFree(out_dims_dev);
        }
        return 163;
    }
    if (rank > 0) {
        hipFree(in_strides_dev);
        hipFree(out_dims_dev);
    }
    return 0;
}

template <typename T>
int binary_broadcast_device(
    int op,
    int device_ordinal,
    int rank,
    size_t total_elems,
    const void* lhs,
    const void* rhs,
    const int* lhs_strides,
    const int* rhs_strides,
    const int* out_dims,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    int* lhs_strides_dev = nullptr;
    int* rhs_strides_dev = nullptr;
    int* out_dims_dev = nullptr;
    const size_t bytes = static_cast<size_t>(rank) * sizeof(int);
    if (hipMalloc(&lhs_strides_dev, bytes) != hipSuccess) return 137;
    if (hipMalloc(&rhs_strides_dev, bytes) != hipSuccess) {
        hipFree(lhs_strides_dev);
        return 137;
    }
    if (hipMalloc(&out_dims_dev, bytes) != hipSuccess) {
        hipFree(lhs_strides_dev);
        hipFree(rhs_strides_dev);
        return 137;
    }
    if (hipMemcpy(lhs_strides_dev, lhs_strides, bytes, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(rhs_strides_dev, rhs_strides, bytes, hipMemcpyHostToDevice) != hipSuccess ||
        hipMemcpy(out_dims_dev, out_dims, bytes, hipMemcpyHostToDevice) != hipSuccess) {
        hipFree(lhs_strides_dev);
        hipFree(rhs_strides_dev);
        hipFree(out_dims_dev);
        return 137;
    }
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((total_elems + static_cast<size_t>(block) - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_binary_broadcast_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        op,
        rank,
        total_elems,
        static_cast<const T*>(lhs),
        static_cast<const T*>(rhs),
        lhs_strides_dev,
        rhs_strides_dev,
        out_dims_dev,
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) {
        hipFree(lhs_strides_dev);
        hipFree(rhs_strides_dev);
        hipFree(out_dims_dev);
        return 138;
    }
    if (maybe_sync() != hipSuccess) {
        hipFree(lhs_strides_dev);
        hipFree(rhs_strides_dev);
        hipFree(out_dims_dev);
        return 139;
    }
    hipFree(lhs_strides_dev);
    hipFree(rhs_strides_dev);
    hipFree(out_dims_dev);
    return 0;
}

template <typename T>
int reduce_keepdim_view_device(
    int device_ordinal,
    int rank,
    int reduce_dim,
    size_t reduce_len,
    size_t total_out_elems,
    int sum,
    const void* xs,
    const int* in_strides,
    const int* out_dims,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    int* in_strides_dev = nullptr;
    int* out_dims_dev = nullptr;
    const size_t bytes = static_cast<size_t>(rank) * sizeof(int);
    if (rank > 0) {
        if (hipMalloc(&in_strides_dev, bytes) != hipSuccess) return 167;
        if (hipMalloc(&out_dims_dev, bytes) != hipSuccess) {
            hipFree(in_strides_dev);
            return 167;
        }
        if (hipMemcpy(in_strides_dev, in_strides, bytes, hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(out_dims_dev, out_dims, bytes, hipMemcpyHostToDevice) != hipSuccess) {
            hipFree(in_strides_dev);
            hipFree(out_dims_dev);
            return 167;
        }
    }
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((total_out_elems + static_cast<size_t>(block) - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_reduce_keepdim_view_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        rank,
        reduce_dim,
        reduce_len,
        total_out_elems,
        sum,
        static_cast<const T*>(xs),
        in_strides_dev,
        out_dims_dev,
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) {
        if (rank > 0) {
            hipFree(in_strides_dev);
            hipFree(out_dims_dev);
        }
        return 168;
    }
    if (maybe_sync() != hipSuccess) {
        if (rank > 0) {
            hipFree(in_strides_dev);
            hipFree(out_dims_dev);
        }
        return 169;
    }
    if (rank > 0) {
        hipFree(in_strides_dev);
        hipFree(out_dims_dev);
    }
    return 0;
}

template <typename T>
int batched_matmul_device(
    int device_ordinal,
    int batch_rank,
    size_t batch_elems,
    int m,
    int n,
    int k,
    const int* lhs_batch_dims,
    const int* rhs_batch_dims,
    const int* out_batch_dims,
    const void* lhs,
    const void* rhs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    int* lhs_batch_dims_dev = nullptr;
    int* rhs_batch_dims_dev = nullptr;
    int* out_batch_dims_dev = nullptr;
    const size_t bytes = static_cast<size_t>(batch_rank) * sizeof(int);
    if (batch_rank > 0) {
        if (hipMalloc(&lhs_batch_dims_dev, bytes) != hipSuccess) return 141;
        if (hipMalloc(&rhs_batch_dims_dev, bytes) != hipSuccess) {
            hipFree(lhs_batch_dims_dev);
            return 141;
        }
        if (hipMalloc(&out_batch_dims_dev, bytes) != hipSuccess) {
            hipFree(lhs_batch_dims_dev);
            hipFree(rhs_batch_dims_dev);
            return 141;
        }
        if (hipMemcpy(lhs_batch_dims_dev, lhs_batch_dims, bytes, hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(rhs_batch_dims_dev, rhs_batch_dims, bytes, hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(out_batch_dims_dev, out_batch_dims, bytes, hipMemcpyHostToDevice) != hipSuccess) {
            hipFree(lhs_batch_dims_dev);
            hipFree(rhs_batch_dims_dev);
            hipFree(out_batch_dims_dev);
            return 141;
        }
    }
    constexpr int block = 256;
    const size_t total = batch_elems * static_cast<size_t>(m) * static_cast<size_t>(n);
    const unsigned int grid =
        static_cast<unsigned int>((total + static_cast<size_t>(block) - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_batched_matmul_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_rank,
        batch_elems,
        m,
        n,
        k,
        lhs_batch_dims_dev,
        rhs_batch_dims_dev,
        out_batch_dims_dev,
        static_cast<const T*>(lhs),
        static_cast<const T*>(rhs),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) {
        if (batch_rank > 0) {
            hipFree(lhs_batch_dims_dev);
            hipFree(rhs_batch_dims_dev);
            hipFree(out_batch_dims_dev);
        }
        return 142;
    }
    if (maybe_sync() != hipSuccess) {
        if (batch_rank > 0) {
            hipFree(lhs_batch_dims_dev);
            hipFree(rhs_batch_dims_dev);
            hipFree(out_batch_dims_dev);
        }
        return 143;
    }
    if (batch_rank > 0) {
        hipFree(lhs_batch_dims_dev);
        hipFree(rhs_batch_dims_dev);
        hipFree(out_batch_dims_dev);
    }
    return 0;
}

template <typename T>
int batched_matmul_view_device(
    int device_ordinal,
    int batch_rank,
    size_t batch_elems,
    int m,
    int n,
    int k,
    const int* lhs_batch_strides,
    const int* rhs_batch_strides,
    const int* out_batch_dims,
    int lhs_row_stride,
    int lhs_k_stride,
    int rhs_k_stride,
    int rhs_col_stride,
    const void* lhs,
    const void* rhs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    int* lhs_batch_strides_dev = nullptr;
    int* rhs_batch_strides_dev = nullptr;
    int* out_batch_dims_dev = nullptr;
    const size_t bytes = static_cast<size_t>(batch_rank) * sizeof(int);
    if (batch_rank > 0) {
        if (hipMalloc(&lhs_batch_strides_dev, bytes) != hipSuccess) return 171;
        if (hipMalloc(&rhs_batch_strides_dev, bytes) != hipSuccess) {
            hipFree(lhs_batch_strides_dev);
            return 171;
        }
        if (hipMalloc(&out_batch_dims_dev, bytes) != hipSuccess) {
            hipFree(lhs_batch_strides_dev);
            hipFree(rhs_batch_strides_dev);
            return 171;
        }
        if (hipMemcpy(lhs_batch_strides_dev, lhs_batch_strides, bytes, hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(rhs_batch_strides_dev, rhs_batch_strides, bytes, hipMemcpyHostToDevice) != hipSuccess ||
            hipMemcpy(out_batch_dims_dev, out_batch_dims, bytes, hipMemcpyHostToDevice) != hipSuccess) {
            hipFree(lhs_batch_strides_dev);
            hipFree(rhs_batch_strides_dev);
            hipFree(out_batch_dims_dev);
            return 171;
        }
    }
    constexpr int block = 256;
    const size_t total = batch_elems * static_cast<size_t>(m) * static_cast<size_t>(n);
    const unsigned int grid =
        static_cast<unsigned int>((total + static_cast<size_t>(block) - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_batched_matmul_view_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_rank,
        batch_elems,
        m,
        n,
        k,
        lhs_batch_strides_dev,
        rhs_batch_strides_dev,
        out_batch_dims_dev,
        lhs_row_stride,
        lhs_k_stride,
        rhs_k_stride,
        rhs_col_stride,
        static_cast<const T*>(lhs),
        static_cast<const T*>(rhs),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) {
        if (batch_rank > 0) {
            hipFree(lhs_batch_strides_dev);
            hipFree(rhs_batch_strides_dev);
            hipFree(out_batch_dims_dev);
        }
        return 172;
    }
    if (maybe_sync() != hipSuccess) {
        if (batch_rank > 0) {
            hipFree(lhs_batch_strides_dev);
            hipFree(rhs_batch_strides_dev);
            hipFree(out_batch_dims_dev);
        }
        return 173;
    }
    if (batch_rank > 0) {
        hipFree(lhs_batch_strides_dev);
        hipFree(rhs_batch_strides_dev);
        hipFree(out_batch_dims_dev);
    }
    return 0;
}

template <typename T>
int mul_scalar_device(
    int device_ordinal,
    int total_elems,
    float scalar,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<size_t>(total_elems) + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_mul_scalar_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        total_elems,
        scalar,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 145;
    if (maybe_sync() != hipSuccess) return 146;
    return 0;
}

template <typename T>
int reduce_keepdim_device(
    int device_ordinal,
    int outer,
    int reduce,
    int inner,
    bool sum,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const int total = outer * inner;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<size_t>(total) + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_reduce_keepdim_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        outer,
        reduce,
        inner,
        sum ? 1 : 0,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 147;
    if (maybe_sync() != hipSuccess) return 148;
    return 0;
}

template <typename T>
int add_scalar_device(
    int device_ordinal,
    int total_elems,
    float scalar,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<size_t>(total_elems) + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_add_scalar_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        total_elems,
        scalar,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 149;
    if (maybe_sync() != hipSuccess) return 150;
    return 0;
}

template <typename T>
int sqrt_device(
    int device_ordinal,
    int total_elems,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<size_t>(total_elems) + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_sqrt_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        total_elems,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 151;
    if (maybe_sync() != hipSuccess) return 152;
    return 0;
}

template <typename T>
int delta_full_scan_pack_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    const void* query_scan,
    const void* key_scan,
    const void* exp_g_scan,
    const void* k_cumdecay_scan,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256 || chunk_size > 64) return 106;
    constexpr int block = 256;
    const size_t total_rows =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(num_chunks) *
        static_cast<size_t>(chunk_size);
    const unsigned int grid = static_cast<unsigned int>((total_rows + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_delta_full_scan_pack_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        static_cast<const T*>(query_scan),
        static_cast<const T*>(key_scan),
        static_cast<const T*>(exp_g_scan),
        static_cast<const T*>(k_cumdecay_scan),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 107;
    if (maybe_sync() != hipSuccess) return 108;
    return 0;
}

template <typename T>
int delta_full_scan_packed_device(
    int device_ordinal,
    int batch_heads,
    int num_chunks,
    int chunk_size,
    int k_head_dim,
    int v_head_dim,
    const void* initial_state,
    const void* packed_scan,
    const void* local_attn_scan,
    const void* value,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (k_head_dim > 256 || chunk_size > 64) return 109;
    constexpr int block = 256;
    const size_t total_threads =
        static_cast<size_t>(batch_heads) * static_cast<size_t>(v_head_dim);
    const unsigned int grid = static_cast<unsigned int>((total_threads + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_delta_full_scan_packed_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        batch_heads,
        num_chunks,
        chunk_size,
        k_head_dim,
        v_head_dim,
        static_cast<const T*>(initial_state),
        static_cast<const T*>(packed_scan),
        static_cast<const T*>(local_attn_scan),
        static_cast<const T*>(value),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 110;
    if (maybe_sync() != hipSuccess) return 111;
    return 0;
}

template <typename T>
int l2norm_device(
    int device_ordinal,
    int n_rows,
    int n_cols,
    float eps,
    const void* xs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_l2norm_kernel<T>),
        dim3(static_cast<unsigned int>(n_rows)),
        dim3(block),
        0,
        0,
        n_rows,
        n_cols,
        eps,
        static_cast<const T*>(xs),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 90;
    if (maybe_sync() != hipSuccess) return 91;
    return 0;
}

template <typename T>
int value_decay_device(
    int device_ordinal,
    int total_elems,
    int num_heads,
    const void* a,
    const void* dt_bias,
    const void* a_log_exp,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const unsigned int grid =
        static_cast<unsigned int>((static_cast<size_t>(total_elems) + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_value_decay_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        total_elems,
        num_heads,
        static_cast<const T*>(a),
        static_cast<const T*>(dt_bias),
        static_cast<const T*>(a_log_exp),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 93;
    if (maybe_sync() != hipSuccess) return 94;
    return 0;
}

template <typename T, bool ADD_UNIT_OFFSET>
int rms_norm_device(
    int device_ordinal,
    int n_rows,
    int n_cols,
    float eps,
    const void* xs,
    const void* weight,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_rms_norm_kernel<T, ADD_UNIT_OFFSET>),
        dim3(static_cast<unsigned int>(n_rows)),
        dim3(block),
        0,
        0,
        n_rows,
        n_cols,
        eps,
        static_cast<const T*>(xs),
        static_cast<const T*>(weight),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 71;
    if (maybe_sync() != hipSuccess) return 72;
    return 0;
}

template <typename T, bool ADD_UNIT_OFFSET>
int fused_rms_norm_linear_device(
    int device_ordinal,
    int hidden_dim,
    int out_dim,
    float eps,
    const void* hidden,
    const void* norm_weight,
    const void* proj_weight,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const size_t shared_bytes =
        static_cast<size_t>(hidden_dim) * sizeof(float) + block * sizeof(float);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_fused_rms_norm_linear_kernel<T, ADD_UNIT_OFFSET>),
        dim3(static_cast<unsigned int>(out_dim)),
        dim3(block),
        shared_bytes,
        0,
        hidden_dim,
        out_dim,
        eps,
        static_cast<const T*>(hidden),
        static_cast<const T*>(norm_weight),
        static_cast<const T*>(proj_weight),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 130;
    if (maybe_sync() != hipSuccess) return 131;
    return 0;
}

template <typename T>
int rms_norm_gated_device(
    int device_ordinal,
    int n_rows,
    int n_cols,
    float eps,
    const void* hidden,
    const void* gate,
    const void* weight,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_rms_norm_gated_kernel<T>),
        dim3(static_cast<unsigned int>(n_rows)),
        dim3(block),
        0,
        0,
        n_rows,
        n_cols,
        eps,
        static_cast<const T*>(hidden),
        static_cast<const T*>(gate),
        static_cast<const T*>(weight),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 81;
    if (maybe_sync() != hipSuccess) return 82;
    return 0;
}

} // namespace

extern "C" int supersonic_qwen35_4b_hip_full_attention_prefill(
    int dtype,
    size_t device_ordinal,
    size_t batch_size,
    size_t q_heads,
    size_t kv_heads,
    size_t q_len,
    size_t kv_len,
    size_t head_dim,
    size_t num_kv_groups,
    float scale,
    size_t seqlen_offset,
    const void* query,
    const void* key,
    const void* value,
    void* out) {
    switch (dtype) {
    case 0:
        return full_attention_prefill_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(q_heads),
            static_cast<int>(kv_heads),
            static_cast<int>(q_len),
            static_cast<int>(kv_len),
            static_cast<int>(head_dim),
            static_cast<int>(num_kv_groups),
            scale,
            static_cast<int>(seqlen_offset),
            query,
            key,
            value,
            out);
    case 1:
        return full_attention_prefill_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(q_heads),
            static_cast<int>(kv_heads),
            static_cast<int>(q_len),
            static_cast<int>(kv_len),
            static_cast<int>(head_dim),
            static_cast<int>(num_kv_groups),
            scale,
            static_cast<int>(seqlen_offset),
            query,
            key,
            value,
            out);
    case 2:
        return full_attention_prefill_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(q_heads),
            static_cast<int>(kv_heads),
            static_cast<int>(q_len),
            static_cast<int>(kv_len),
            static_cast<int>(head_dim),
            static_cast<int>(num_kv_groups),
            scale,
            static_cast<int>(seqlen_offset),
            query,
            key,
            value,
            out);
    default:
        return 64;
    }
}

extern "C" int supersonic_qwen35_4b_hip_linear_prefill_conv_pack(
    int dtype,
    size_t device_ordinal,
    size_t batch_size,
    size_t conv_dim,
    size_t total_len,
    size_t seq_len,
    size_t kernel_size,
    const void* mixed_qkv,
    const void* weights,
    void* out) {
    switch (dtype) {
    case 0:
        return linear_prefill_conv_pack_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(total_len),
            static_cast<int>(seq_len),
            static_cast<int>(kernel_size),
            mixed_qkv,
            weights,
            out);
    case 1:
        return linear_prefill_conv_pack_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(total_len),
            static_cast<int>(seq_len),
            static_cast<int>(kernel_size),
            mixed_qkv,
            weights,
            out);
    case 2:
        return linear_prefill_conv_pack_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(conv_dim),
            static_cast<int>(total_len),
            static_cast<int>(seq_len),
            static_cast<int>(kernel_size),
            mixed_qkv,
            weights,
            out);
    default:
        return 62;
    }
}

extern "C" int supersonic_qwen35_4b_hip_delta_recurrent_prefill(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t seq_len,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* initial_state,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_recurrent_prefill_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(seq_len),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    case 1:
        return delta_recurrent_prefill_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(seq_len),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    case 2:
        return delta_recurrent_prefill_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(seq_len),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    default:
        return 66;
    }
}

extern "C" int supersonic_qwen35_4b_hip_delta_chunk_single_prefill(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_chunk_single_prefill_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            query,
            key,
            value,
            beta,
            g,
            out);
    case 1:
        return delta_chunk_single_prefill_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            query,
            key,
            value,
            beta,
            g,
            out);
    case 2:
        return delta_chunk_single_prefill_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            query,
            key,
            value,
            beta,
            g,
            out);
    default:
        return 79;
    }
}

extern "C" int supersonic_qwen35_4b_hip_delta_chunk_step(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* prev_state,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_chunk_step_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            prev_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    case 1:
        return delta_chunk_step_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            prev_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    case 2:
        return delta_chunk_step_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            prev_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    default:
        return 86;
    }
}

extern "C" int supersonic_qwen35_4b_hip_delta_chunk_scan_raw(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* initial_state,
    const void* query,
    const void* key,
    const void* value,
    const void* beta,
    const void* g,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_chunk_scan_raw_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    case 1:
        return delta_chunk_scan_raw_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    case 2:
        return delta_chunk_scan_raw_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            query,
            key,
            value,
            beta,
            g,
            out);
    default:
        return 87;
    }
}

extern "C" int supersonic_qwen35_4b_hip_delta_state_scan(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* initial_state,
    const void* packed_scan,
    const void* value,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_state_scan_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            packed_scan,
            value,
            out);
    case 1:
        return delta_state_scan_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            packed_scan,
            value,
            out);
    case 2:
        return delta_state_scan_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            packed_scan,
            value,
            out);
    default:
        return 103;
    }
}

extern "C" int supersonic_qwen35_4b_hip_delta_chunk_fused(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* prev_state,
    const void* packed_chunk,
    const void* value,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_chunk_fused_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            prev_state,
            packed_chunk,
            value,
            out);
    case 1:
        return delta_chunk_fused_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            prev_state,
            packed_chunk,
            value,
            out);
    case 2:
        return delta_chunk_fused_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            prev_state,
            packed_chunk,
            value,
            out);
    default:
        return 104;
    }
}

extern "C" int supersonic_qwen35_4b_hip_delta_full_scan(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* initial_state,
    const void* weighted_key_scan,
    const void* k_cumdecay_scan,
    const void* q_state_scan,
    const void* local_attn_scan,
    const void* state_decay_scan,
    const void* value,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_full_scan_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            weighted_key_scan,
            k_cumdecay_scan,
            q_state_scan,
            local_attn_scan,
            state_decay_scan,
            value,
            out);
    case 1:
        return delta_full_scan_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            weighted_key_scan,
            k_cumdecay_scan,
            q_state_scan,
            local_attn_scan,
            state_decay_scan,
            value,
            out);
    case 2:
        return delta_full_scan_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            weighted_key_scan,
            k_cumdecay_scan,
            q_state_scan,
            local_attn_scan,
            state_decay_scan,
            value,
            out);
    default:
        return 105;
    }
}

extern "C" int supersonic_qwen35_4b_hip_delta_full_scan_pack(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    const void* query_scan,
    const void* key_scan,
    const void* exp_g_scan,
    const void* k_cumdecay_scan,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_full_scan_pack_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            query_scan,
            key_scan,
            exp_g_scan,
            k_cumdecay_scan,
            out);
    case 1:
        return delta_full_scan_pack_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            query_scan,
            key_scan,
            exp_g_scan,
            k_cumdecay_scan,
            out);
    case 2:
        return delta_full_scan_pack_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            query_scan,
            key_scan,
            exp_g_scan,
            k_cumdecay_scan,
            out);
    default:
        return 112;
    }
}

extern "C" int supersonic_qwen35_4b_hip_delta_local_attn_scan(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    const void* query_scan,
    const void* key_scan,
    const void* exp_g_scan,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_local_attn_scan_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            query_scan,
            key_scan,
            exp_g_scan,
            out);
    case 1:
        return delta_local_attn_scan_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            query_scan,
            key_scan,
            exp_g_scan,
            out);
    case 2:
        return delta_local_attn_scan_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            query_scan,
            key_scan,
            exp_g_scan,
            out);
    default:
        return 114;
    }
}

extern "C" int supersonic_qwen35_4b_hip_delta_base_attn_scan(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    const void* k_beta_scan,
    const void* key_scan,
    const void* exp_g_scan,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_base_attn_scan_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            k_beta_scan,
            key_scan,
            exp_g_scan,
            out);
    case 1:
        return delta_base_attn_scan_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            k_beta_scan,
            key_scan,
            exp_g_scan,
            out);
    case 2:
        return delta_base_attn_scan_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            k_beta_scan,
            key_scan,
            exp_g_scan,
            out);
    default:
        return 117;
    }
}

extern "C" int supersonic_qwen35_4b_hip_delta_attn_solve_scan(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    const void* base_attn_scan,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_attn_solve_scan_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            base_attn_scan,
            out);
    case 1:
        return delta_attn_solve_scan_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            base_attn_scan,
            out);
    case 2:
        return delta_attn_solve_scan_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            base_attn_scan,
            out);
    default:
        return 120;
    }
}

extern "C" int supersonic_qwen35_4b_hip_delta_attn_solve_from_inputs(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    const void* k_beta_scan,
    const void* key_scan,
    const void* exp_g_scan,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_attn_solve_from_inputs_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            k_beta_scan,
            key_scan,
            exp_g_scan,
            out);
    case 1:
        return delta_attn_solve_from_inputs_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            k_beta_scan,
            key_scan,
            exp_g_scan,
            out);
    case 2:
        return delta_attn_solve_from_inputs_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            k_beta_scan,
            key_scan,
            exp_g_scan,
            out);
    default:
        return 123;
    }
}

extern "C" int supersonic_qwen35_4b_hip_swiglu_mul(
    int dtype,
    size_t device_ordinal,
    size_t elem_count,
    const void* gate,
    const void* up,
    void* out) {
    switch (dtype) {
    case 0:
        return swiglu_mul_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(elem_count),
            gate,
            up,
            out);
    case 1:
        return swiglu_mul_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(elem_count),
            gate,
            up,
            out);
    case 2:
        return swiglu_mul_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(elem_count),
            gate,
            up,
            out);
    default:
        return 122;
    }
}

extern "C" int supersonic_qwen35_4b_hip_embedding_lookup(
    int dtype,
    int index_dtype,
    size_t device_ordinal,
    size_t token_count,
    size_t vocab_size,
    size_t hidden_size,
    const void* embeddings,
    const void* indexes,
    void* out) {
    switch (dtype) {
    case 0:
        switch (index_dtype) {
        case 0:
            return embedding_lookup_device<half, uint8_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        case 1:
            return embedding_lookup_device<half, uint32_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        case 2:
            return embedding_lookup_device<half, int64_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        default:
            return 123;
        }
    case 1:
        switch (index_dtype) {
        case 0:
            return embedding_lookup_device<float, uint8_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        case 1:
            return embedding_lookup_device<float, uint32_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        case 2:
            return embedding_lookup_device<float, int64_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        default:
            return 123;
        }
    case 2:
        switch (index_dtype) {
        case 0:
            return embedding_lookup_device<hip_bfloat16, uint8_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        case 1:
            return embedding_lookup_device<hip_bfloat16, uint32_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        case 2:
            return embedding_lookup_device<hip_bfloat16, int64_t>(
                static_cast<int>(device_ordinal),
                static_cast<int>(token_count),
                static_cast<int>(vocab_size),
                static_cast<int>(hidden_size),
                embeddings,
                indexes,
                out);
        default:
            return 123;
        }
    default:
        return 124;
    }
}

template <typename T>
int output_projection_lookup_device(
    int device_ordinal,
    int rows,
    int hidden_size,
    int vocab_size,
    const void* hidden,
    const void* weights,
    void* out) {
    ScopedHipDevice scoped(device_ordinal);
    const int total_elems = rows * vocab_size;
    const int block = 256;
    const int grid = (total_elems + block - 1) / block;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_output_projection_lookup_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        rows,
        hidden_size,
        vocab_size,
        static_cast<const T*>(hidden),
        static_cast<const T*>(weights),
        static_cast<T*>(out));
    if (hipGetLastError() != hipSuccess) return 11;
    return 0;
}

extern "C" int supersonic_qwen35_4b_hip_output_projection_lookup(
    int dtype,
    size_t device_ordinal,
    size_t rows,
    size_t hidden_size,
    size_t vocab_size,
    const void* hidden,
    const void* weights,
    void* out) {
    switch (dtype) {
    case 0:
        return output_projection_lookup_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(rows),
            static_cast<int>(hidden_size),
            static_cast<int>(vocab_size),
            hidden,
            weights,
            out);
    case 1:
        return output_projection_lookup_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(rows),
            static_cast<int>(hidden_size),
            static_cast<int>(vocab_size),
            hidden,
            weights,
            out);
    case 2:
        return output_projection_lookup_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(rows),
            static_cast<int>(hidden_size),
            static_cast<int>(vocab_size),
            hidden,
            weights,
            out);
    default:
        return 122;
    }
}

extern "C" int supersonic_qwen35_4b_hip_causal_mask(
    int dtype,
    size_t device_ordinal,
    size_t batch_size,
    size_t tgt_len,
    size_t seqlen_offset,
    void* out) {
    switch (dtype) {
    case 0:
        return causal_mask_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(tgt_len),
            static_cast<int>(seqlen_offset),
            out);
    case 1:
        return causal_mask_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(tgt_len),
            static_cast<int>(seqlen_offset),
            out);
    case 2:
        return causal_mask_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_size),
            static_cast<int>(tgt_len),
            static_cast<int>(seqlen_offset),
            out);
    default:
        return 126;
    }
}

extern "C" int supersonic_qwen35_4b_hip_cumsum_last_dim(
    int dtype,
    size_t device_ordinal,
    size_t rows,
    size_t cols,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return cumsum_last_dim_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(rows),
            static_cast<int>(cols),
            xs,
            out);
    case 1:
        return cumsum_last_dim_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(rows),
            static_cast<int>(cols),
            xs,
            out);
    case 2:
        return cumsum_last_dim_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(rows),
            static_cast<int>(cols),
            xs,
            out);
    default:
        return 128;
    }
}

extern "C" int supersonic_qwen35_4b_hip_delta_full_scan_packed(
    int dtype,
    size_t device_ordinal,
    size_t batch_heads,
    size_t num_chunks,
    size_t chunk_size,
    size_t k_head_dim,
    size_t v_head_dim,
    const void* initial_state,
    const void* packed_scan,
    const void* local_attn_scan,
    const void* value,
    void* out) {
    switch (dtype) {
    case 0:
        return delta_full_scan_packed_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            packed_scan,
            local_attn_scan,
            value,
            out);
    case 1:
        return delta_full_scan_packed_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            packed_scan,
            local_attn_scan,
            value,
            out);
    case 2:
        return delta_full_scan_packed_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(batch_heads),
            static_cast<int>(num_chunks),
            static_cast<int>(chunk_size),
            static_cast<int>(k_head_dim),
            static_cast<int>(v_head_dim),
            initial_state,
            packed_scan,
            local_attn_scan,
            value,
            out);
    default:
        return 113;
    }
}

extern "C" int supersonic_qwen35_4b_hip_exp(
    int dtype,
    size_t device_ordinal,
    size_t total_elems,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return exp_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    case 1:
        return exp_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    case 2:
        return exp_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    default:
        return 129;
    }
}

extern "C" int supersonic_qwen35_4b_hip_recip(
    int dtype,
    size_t device_ordinal,
    size_t total_elems,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return recip_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    case 1:
        return recip_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    case 2:
        return recip_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    default:
        return 131;
    }
}

extern "C" int supersonic_qwen35_4b_hip_sigmoid(
    int dtype,
    size_t device_ordinal,
    size_t total_elems,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return sigmoid_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    case 1:
        return sigmoid_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    case 2:
        return sigmoid_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    default:
        return 133;
    }
}

extern "C" int supersonic_qwen35_4b_hip_log(
    int dtype,
    size_t device_ordinal,
    size_t total_elems,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return log_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    case 1:
        return log_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    case 2:
        return log_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    default:
        return 157;
    }
}

extern "C" int supersonic_qwen35_4b_hip_unary_view(
    int op,
    int dtype,
    size_t device_ordinal,
    int rank,
    size_t total_elems,
    float scalar,
    const void* xs,
    const int* in_strides,
    const int* out_dims,
    void* out) {
    switch (dtype) {
    case 0:
        return unary_view_device<half>(
            op, static_cast<int>(device_ordinal), rank, total_elems, scalar, xs, in_strides, out_dims, out);
    case 1:
        return unary_view_device<float>(
            op, static_cast<int>(device_ordinal), rank, total_elems, scalar, xs, in_strides, out_dims, out);
    case 2:
        return unary_view_device<hip_bfloat16>(
            op, static_cast<int>(device_ordinal), rank, total_elems, scalar, xs, in_strides, out_dims, out);
    default:
        return 164;
    }
}

extern "C" int supersonic_qwen35_4b_hip_cast_view(
    int input_dtype,
    int output_dtype,
    size_t device_ordinal,
    int rank,
    size_t total_elems,
    const void* xs,
    const int* in_strides,
    const int* out_dims,
    void* out) {
    switch (input_dtype) {
    case 0:
        switch (output_dtype) {
        case 0:
            return cast_view_device<half, half>(static_cast<int>(device_ordinal), rank, total_elems, xs, in_strides, out_dims, out);
        case 1:
            return cast_view_device<half, float>(static_cast<int>(device_ordinal), rank, total_elems, xs, in_strides, out_dims, out);
        case 2:
            return cast_view_device<half, hip_bfloat16>(static_cast<int>(device_ordinal), rank, total_elems, xs, in_strides, out_dims, out);
        default:
            return 165;
        }
    case 1:
        switch (output_dtype) {
        case 0:
            return cast_view_device<float, half>(static_cast<int>(device_ordinal), rank, total_elems, xs, in_strides, out_dims, out);
        case 1:
            return cast_view_device<float, float>(static_cast<int>(device_ordinal), rank, total_elems, xs, in_strides, out_dims, out);
        case 2:
            return cast_view_device<float, hip_bfloat16>(static_cast<int>(device_ordinal), rank, total_elems, xs, in_strides, out_dims, out);
        default:
            return 165;
        }
    case 2:
        switch (output_dtype) {
        case 0:
            return cast_view_device<hip_bfloat16, half>(static_cast<int>(device_ordinal), rank, total_elems, xs, in_strides, out_dims, out);
        case 1:
            return cast_view_device<hip_bfloat16, float>(static_cast<int>(device_ordinal), rank, total_elems, xs, in_strides, out_dims, out);
        case 2:
            return cast_view_device<hip_bfloat16, hip_bfloat16>(static_cast<int>(device_ordinal), rank, total_elems, xs, in_strides, out_dims, out);
        default:
            return 165;
        }
    default:
        return 166;
    }
}

extern "C" int supersonic_qwen35_4b_hip_reduce_keepdim_view(
    int dtype,
    size_t device_ordinal,
    int rank,
    int reduce_dim,
    size_t reduce_len,
    size_t total_out_elems,
    int sum,
    const void* xs,
    const int* in_strides,
    const int* out_dims,
    void* out) {
    switch (dtype) {
    case 0:
        return reduce_keepdim_view_device<half>(
            static_cast<int>(device_ordinal), rank, reduce_dim, reduce_len, total_out_elems, sum, xs, in_strides, out_dims, out);
    case 1:
        return reduce_keepdim_view_device<float>(
            static_cast<int>(device_ordinal), rank, reduce_dim, reduce_len, total_out_elems, sum, xs, in_strides, out_dims, out);
    case 2:
        return reduce_keepdim_view_device<hip_bfloat16>(
            static_cast<int>(device_ordinal), rank, reduce_dim, reduce_len, total_out_elems, sum, xs, in_strides, out_dims, out);
    default:
        return 170;
    }
}

extern "C" int supersonic_qwen35_4b_hip_batched_matmul_view(
    int dtype,
    size_t device_ordinal,
    int batch_rank,
    size_t batch_elems,
    int m,
    int n,
    int k,
    const int* lhs_batch_strides,
    const int* rhs_batch_strides,
    const int* out_batch_dims,
    int lhs_row_stride,
    int lhs_k_stride,
    int rhs_k_stride,
    int rhs_col_stride,
    const void* lhs,
    const void* rhs,
    void* out) {
    switch (dtype) {
    case 0:
        return batched_matmul_view_device<half>(
            static_cast<int>(device_ordinal), batch_rank, batch_elems, m, n, k,
            lhs_batch_strides, rhs_batch_strides, out_batch_dims,
            lhs_row_stride, lhs_k_stride, rhs_k_stride, rhs_col_stride, lhs, rhs, out);
    case 1:
        return batched_matmul_view_device<float>(
            static_cast<int>(device_ordinal), batch_rank, batch_elems, m, n, k,
            lhs_batch_strides, rhs_batch_strides, out_batch_dims,
            lhs_row_stride, lhs_k_stride, rhs_k_stride, rhs_col_stride, lhs, rhs, out);
    case 2:
        return batched_matmul_view_device<hip_bfloat16>(
            static_cast<int>(device_ordinal), batch_rank, batch_elems, m, n, k,
            lhs_batch_strides, rhs_batch_strides, out_batch_dims,
            lhs_row_stride, lhs_k_stride, rhs_k_stride, rhs_col_stride, lhs, rhs, out);
    default:
        return 174;
    }
}

// Tiled BF16 matmul for prefill: out = lhs × rhs^T (rhs stored [n, k])
template <typename T>
int matmul_rhs_transposed_tiled_device(
    int device_ordinal,
    size_t batch_elems,
    int m, int n, int k,
    const void* lhs,
    const void* rhs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int TILE_M = 16;
    constexpr int TILE_N = 16;
    const int grid_x = (n + TILE_N - 1) / TILE_N;
    const int grid_y = (m + TILE_M - 1) / TILE_M;
    const int grid_z = static_cast<int>(batch_elems);
    const int threads = TILE_M * TILE_N;  // 256
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_matmul_rhs_transposed_tiled_kernel<T>),
        dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
        batch_elems, m, n, k,
        static_cast<const T*>(lhs),
        static_cast<const T*>(rhs),
        static_cast<T*>(out));
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err = maybe_sync();
    if (launch_err != hipSuccess) return prefill_backend_failure(270, launch_err);
    if (sync_err != hipSuccess) return prefill_backend_failure(271, sync_err);
    return 0;
}

// Cached per-device arch flag: does this device support the validated RDNA WMMA
// path? gfx11xx (RDNA3 / RDNA3.5) supports the original wave32 BF16/i8 layout.
// gfx12xx (RDNA4) uses the new `_gfx12` operand layout, adapted in
// full_attention_4b.hip.
//
// Env var SUPERSONIC_QWEN4B_DISABLE_WMMA=1 forces the scalar path for debugging
// / perf comparison.
//
// `supersonic-serve` can call this concurrently from multiple request threads,
// so initialization goes through `std::call_once` — plain non-atomic writes
// would be a data race.
static bool device_supports_wmma_bf16(int device_ordinal) {
    static std::once_flag env_once;
    static bool env_disabled = false;
    std::call_once(env_once, [] {
        const char* env = std::getenv("SUPERSONIC_QWEN4B_DISABLE_WMMA");
        env_disabled = (env != nullptr && env[0] != '\0' && env[0] != '0');
    });
    if (env_disabled) return false;

    auto probe_arch = [](int ordinal) -> bool {
        hipDeviceProp_t props;
        if (hipGetDeviceProperties(&props, ordinal) != hipSuccess) return false;
        const char* arch = props.gcnArchName;
        if (!arch || arch[0] != 'g' || arch[1] != 'f' || arch[2] != 'x' ||
            arch[3] != '1') {
            return false;
        }
        if (arch[4] == '1') return true;
        return arch[4] == '2';
    };

    if (device_ordinal < 0 || device_ordinal >= 16) {
        // Uncached lookup for unusual ordinals — happens at most once per call
        // for a device outside the cached range.
        return probe_arch(device_ordinal);
    }

    static std::once_flag device_once[16];
    static bool cached[16] = {false};
    std::call_once(device_once[device_ordinal], [&] {
        cached[device_ordinal] = probe_arch(device_ordinal);
    });
    return cached[device_ordinal];
}

static bool device_supports_wmma_i8(int device_ordinal) {
    // gfx12 i8 WMMA drops the gfx11 operand replication and uses a narrower
    // packed A/B operand. The device code adapts the existing four-word packed
    // fragments to the gfx12 two-word form.
    if (!device_supports_wmma_bf16(device_ordinal)) return false;

    auto probe_arch = [](int ordinal) -> bool {
        hipDeviceProp_t props;
        if (hipGetDeviceProperties(&props, ordinal) != hipSuccess) return false;
        const char* arch = props.gcnArchName;
        return arch && arch[0] == 'g' && arch[1] == 'f' && arch[2] == 'x' &&
               arch[3] == '1' && (arch[4] == '1' || arch[4] == '2');
    };

    if (device_ordinal < 0 || device_ordinal >= 16) {
        return probe_arch(device_ordinal);
    }

    static std::once_flag device_once[16];
    static bool cached[16] = {false};
    std::call_once(device_once[device_ordinal], [&] {
        cached[device_ordinal] = probe_arch(device_ordinal);
    });
    return cached[device_ordinal];
}

bool device_is_gfx12(int device_ordinal) {
    auto probe_arch = [](int ordinal) -> bool {
        hipDeviceProp_t props;
        if (hipGetDeviceProperties(&props, ordinal) != hipSuccess) return false;
        const char* arch = props.gcnArchName;
        return arch && arch[0] == 'g' && arch[1] == 'f' && arch[2] == 'x' &&
               arch[3] == '1' && arch[4] == '2';
    };

    if (device_ordinal < 0 || device_ordinal >= 16) {
        return probe_arch(device_ordinal);
    }

    static std::once_flag device_once[16];
    static bool cached[16] = {false};
    std::call_once(device_once[device_ordinal], [&] {
        cached[device_ordinal] = probe_arch(device_ordinal);
    });
    return cached[device_ordinal];
}

static int matmul_rhs_transposed_tiled_wmma_bf16_device(
    int device_ordinal,
    size_t batch_elems,
    int m, int n, int k,
    const void* lhs,
    const void* rhs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    // Must match the TILED_WMMA_B{M,N} constants in full_attention_4b.hip.
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;
    const int grid_x = (n + TILE_N - 1) / TILE_N;
    const int grid_y = (m + TILE_M - 1) / TILE_M;
    const int grid_z = static_cast<int>(batch_elems);
    const int threads = 128;  // 4 wavefronts per block, arranged 2x2
    hipLaunchKernelGGL(
        supersonic_qwen35_matmul_rhs_transposed_tiled_wmma_kernel,
        dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
        batch_elems, m, n, k,
        static_cast<const hip_bfloat16*>(lhs),
        static_cast<const hip_bfloat16*>(rhs),
        static_cast<hip_bfloat16*>(out));
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err = maybe_sync();
    if (launch_err != hipSuccess) {
        return supersonic_qwen35_4b_bf16_matmul_bridge_status(
            280, static_cast<int>(launch_err));
    }
    if (sync_err != hipSuccess) {
        return supersonic_qwen35_4b_bf16_matmul_bridge_status(
            281, static_cast<int>(sync_err));
    }
    return 0;
}

static int matmul_rhs_transposed_wmma_small_m_bf16_device(
    int device_ordinal,
    size_t batch_elems,
    int m, int n, int k,
    const void* lhs,
    const void* rhs,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int TILE_M = 16;
    constexpr int TILE_N = 16;
    const int grid_x = (n + TILE_N - 1) / TILE_N;
    const int grid_y = (m + TILE_M - 1) / TILE_M;
    const int grid_z = static_cast<int>(batch_elems);
    if (device_is_gfx12(device_ordinal)) {
        const bool hot_exact =
            m == TILE_M && (n % TILE_N) == 0 && (k % 16) == 0;
        if (hot_exact) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(supersonic_qwen35_matmul_rhs_transposed_wmma_small_m_gfx12_kernel<true>),
                dim3(grid_x, grid_y, grid_z), dim3(32), 0, 0,
                batch_elems, m, n, k,
                static_cast<const hip_bfloat16*>(lhs),
                static_cast<const hip_bfloat16*>(rhs),
                static_cast<hip_bfloat16*>(out));
        } else {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(supersonic_qwen35_matmul_rhs_transposed_wmma_small_m_gfx12_kernel<false>),
                dim3(grid_x, grid_y, grid_z), dim3(32), 0, 0,
                batch_elems, m, n, k,
                static_cast<const hip_bfloat16*>(lhs),
                static_cast<const hip_bfloat16*>(rhs),
                static_cast<hip_bfloat16*>(out));
        }
    } else {
        hipLaunchKernelGGL(
            supersonic_qwen35_matmul_rhs_transposed_wmma_small_m_kernel,
            dim3(grid_x, grid_y, grid_z), dim3(32), 0, 0,
            batch_elems, m, n, k,
            static_cast<const hip_bfloat16*>(lhs),
            static_cast<const hip_bfloat16*>(rhs),
            static_cast<hip_bfloat16*>(out));
    }
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err = maybe_sync();
    if (launch_err != hipSuccess) {
        return supersonic_qwen35_4b_bf16_matmul_bridge_status(
            282, static_cast<int>(launch_err));
    }
    if (sync_err != hipSuccess) {
        return supersonic_qwen35_4b_bf16_matmul_bridge_status(
            283, static_cast<int>(sync_err));
    }
    return 0;
}

static hipblasHandle_t hipblas_handle_for(int device_ordinal) {
    DecodeBridgeLockGuard guard;
    static hipblasHandle_t handles[16] = {};
    static bool ready[16] = {};
    if (device_ordinal < 0 || device_ordinal >= 16) {
        return nullptr;
    }
    if (!ready[device_ordinal]) {
        if (hipblasCreate(&handles[device_ordinal]) != HIPBLAS_STATUS_SUCCESS) {
            handles[device_ordinal] = nullptr;
        }
        ready[device_ordinal] = true;
    }
    return handles[device_ordinal];
}

// Row-major C[m,n] = A[m,k] * B[n,k]^T via hipBLAS col-major GemmEx.
// Returns 0 on success; non-zero means the caller should use WMMA.
static int matmul_rhs_transposed_hipblas_bf16(
    int device_ordinal,
    size_t batch_elems,
    int m,
    int n,
    int k,
    const void* lhs,
    const void* rhs,
    void* out
) {
    static const bool disabled = [] {
        const char* e = std::getenv("SUPERSONIC_HIPBLAS");
        return e != nullptr && e[0] == '0';
    }();
    if (disabled || batch_elems != 1 || m < 64 || n < 64 || k < 64) {
        return 1;
    }
    hipblasHandle_t handle = hipblas_handle_for(device_ordinal);
    if (handle == nullptr) {
        return 1;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const hipblasStatus_t st = hipblasGemmEx(
        handle,
        HIPBLAS_OP_T,
        HIPBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        rhs,
        HIP_R_16BF,
        k,
        lhs,
        HIP_R_16BF,
        k,
        &beta,
        out,
        HIP_R_16BF,
        n,
        HIPBLAS_COMPUTE_32F,
        HIPBLAS_GEMM_DEFAULT);
    return st == HIPBLAS_STATUS_SUCCESS ? 0 : 1;
}

static hipStream_t gqh_s_dq = nullptr;
static hipStream_t gqh_s_gemm = nullptr;
static hipEvent_t gqh_ev_gemm[2] = {nullptr, nullptr};
static bool gqh_ev_gemm_recorded[2] = {false, false};
static int gqh_resource_device = -1;

struct GqhPendingWorkGuard {
    int device_ordinal;
    bool pending = false;

    explicit GqhPendingWorkGuard(int ordinal) : device_ordinal(ordinal) {}

    void mark() { pending = true; }
    void disarm() { pending = false; }

    void synchronize_or_fail(const char* operation) {
        if (!pending) {
            return;
        }
        const hipError_t err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                operation, static_cast<int>(err), device_ordinal);
        }
        pending = false;
    }

    ~GqhPendingWorkGuard() {
        if (pending) {
            supersonic_gpu_integrity_fail_stop(
                "GQH dequant pending work escaped", static_cast<int>(hipErrorUnknown),
                device_ordinal);
        }
    }
};

extern "C" int supersonic_gqh_hip_gemm_flush(int device_ordinal) {
    DecodeBridgeLockGuard guard;
    const int owner = gqh_resource_device >= 0 ? gqh_resource_device : device_ordinal;
    bool pending = false;
    for (int i = 0; i < 2; ++i) {
        pending = pending || (gqh_ev_gemm_recorded[i] && gqh_ev_gemm[i] != nullptr);
    }
    ScopedHipDevice scoped(owner);
    if (!scoped.ok()) {
        if (pending) {
            supersonic_gpu_integrity_fail_stop(
                "GQH GEMM flush owner switch", static_cast<int>(scoped.status), owner);
        }
        return static_cast<int>(scoped.status);
    }
    auto finish = [&](int status) {
        const hipError_t restore_err = scoped.restore();
        if (restore_err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "GQH GEMM owner restore", static_cast<int>(restore_err), owner);
        }
        return status;
    };
    // Enqueue a default-stream wait so later default-stream kernels see
    // GEMM output, without blocking the CPU (host EventSynchronize was
    // serializing the whole prefill around each consumer). Events are created
    // on `gqh_resource_device`; wait on that owner even when the caller's
    // target ordinal differs, then restore the caller's device on scope exit.
    for (int i = 0; i < 2; ++i) {
        if (gqh_ev_gemm_recorded[i] && gqh_ev_gemm[i] != nullptr) {
            const hipError_t err = hipStreamWaitEvent(nullptr, gqh_ev_gemm[i], 0);
            if (err != hipSuccess) {
                const hipError_t sync_err = hipDeviceSynchronize();
                if (sync_err != hipSuccess) {
                    supersonic_gpu_integrity_fail_stop(
                        "GQH GEMM flush recovery synchronize",
                        static_cast<int>(sync_err),
                        owner);
                }
                return finish(static_cast<int>(err));
            }
        }
    }
    return finish(0);
}

extern "C" int supersonic_gqh_hip_decode(
    int device_ordinal,
    int rung,
    const void* wire,
    float tensor_scale,
    int grid_code,
    void* dst,
    int rows,
    int cols,
    int dst_is_bf16,
    void* stream);

// Pipeline GQH dequant on one stream and hipBLAS GEMM on another with
// double-buffered BF16 weights so consecutive prefills overlap.
extern "C" int supersonic_gqh_hip_dequant_gemm_bf16(
    int device_ordinal,
    int rung,
    const void* wire,
    float tensor_scale,
    int grid_code,
    const void* lhs,
    void* out,
    int m,
    int n,
    int k) {
    DecodeBridgeLockGuard guard;
    if (wire == nullptr || lhs == nullptr || out == nullptr || m <= 0 || n <= 0 ||
        k <= 0) {
        return 1;
    }
    const bool retained_resources = gqh_resource_device >= 0 &&
        (gqh_s_dq != nullptr || gqh_s_gemm != nullptr || gqh_ev_gemm[0] != nullptr ||
         gqh_ev_gemm[1] != nullptr);
    ScopedHipDevice scoped(device_ordinal);
    if (!scoped.ok()) {
        if (retained_resources) {
            supersonic_gpu_integrity_fail_stop(
                "GQH dequant owner switch", static_cast<int>(scoped.status), device_ordinal);
        }
        return 2;
    }
    auto finish = [&](int status) {
        const hipError_t restore_err = scoped.restore();
        if (restore_err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "GQH dequant owner restore", static_cast<int>(restore_err), device_ordinal);
        }
        return status;
    };
    hipblasHandle_t handle = hipblas_handle_for(device_ordinal);
    if (handle == nullptr) {
        return finish(1);
    }
    static hip_bfloat16* buf[2] = {nullptr, nullptr};
    static size_t cap[2] = {0, 0};
    static hipEvent_t ev_dq[2] = {nullptr, nullptr};
    static int phase = 0;
    static bool ready = false;
    auto destroy_resources = [&]() -> hipError_t {
        const int resource_owner =
            gqh_resource_device >= 0 ? gqh_resource_device : device_ordinal;
        ready = false;
        if (gqh_s_dq == nullptr && gqh_s_gemm == nullptr &&
            ev_dq[0] == nullptr && ev_dq[1] == nullptr &&
            gqh_ev_gemm[0] == nullptr && gqh_ev_gemm[1] == nullptr &&
            buf[0] == nullptr && buf[1] == nullptr) {
            ready = false;
            phase = 0;
            gqh_resource_device = -1;
            return hipSuccess;
        }
        hipError_t err = hipDeviceSynchronize();
        if (err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "GQH dequant resource synchronize", static_cast<int>(err), resource_owner);
        }
        for (int i = 0; i < 2; ++i) {
            if (ev_dq[i] != nullptr) {
                err = hipEventDestroy(ev_dq[i]);
                if (err != hipSuccess) {
                    supersonic_gpu_integrity_fail_stop(
                        "GQH dequant event destroy", static_cast<int>(err), resource_owner);
                }
                ev_dq[i] = nullptr;
            }
            if (gqh_ev_gemm[i] != nullptr) {
                err = hipEventDestroy(gqh_ev_gemm[i]);
                if (err != hipSuccess) {
                    supersonic_gpu_integrity_fail_stop(
                        "GQH GEMM event destroy", static_cast<int>(err), resource_owner);
                }
                gqh_ev_gemm[i] = nullptr;
            }
            gqh_ev_gemm_recorded[i] = false;
            if (buf[i] != nullptr) {
                err = hipFree(buf[i]);
                if (err != hipSuccess) {
                    supersonic_gpu_integrity_fail_stop(
                        "GQH dequant buffer free", static_cast<int>(err), resource_owner);
                }
                buf[i] = nullptr;
                cap[i] = 0;
            }
        }
        if (gqh_s_dq != nullptr) {
            err = hipStreamDestroy(gqh_s_dq);
            if (err != hipSuccess) {
                supersonic_gpu_integrity_fail_stop(
                    "GQH dequant stream destroy", static_cast<int>(err), resource_owner);
            }
            gqh_s_dq = nullptr;
        }
        if (gqh_s_gemm != nullptr) {
            err = hipStreamDestroy(gqh_s_gemm);
            if (err != hipSuccess) {
                supersonic_gpu_integrity_fail_stop(
                    "GQH GEMM stream destroy", static_cast<int>(err), resource_owner);
            }
            gqh_s_gemm = nullptr;
        }
        ready = false;
        phase = 0;
        gqh_resource_device = -1;
        return hipSuccess;
    };
    if (gqh_resource_device >= 0 && gqh_resource_device != device_ordinal) {
        const int old_device = gqh_resource_device;
        ScopedHipDevice old_owner(old_device);
        if (!old_owner.ok()) {
            supersonic_gpu_integrity_fail_stop(
                "GQH old-owner switch", static_cast<int>(old_owner.status), old_device);
        }
        const hipError_t err = destroy_resources();
        if (err != hipSuccess) {
            return finish(static_cast<int>(err));
        }
        const hipError_t restore_err = old_owner.restore();
        if (restore_err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "GQH old-owner restore", static_cast<int>(restore_err), old_device);
        }
    }
    if (!ready && gqh_resource_device == device_ordinal &&
        (gqh_s_dq != nullptr || gqh_s_gemm != nullptr || ev_dq[0] != nullptr ||
         ev_dq[1] != nullptr || gqh_ev_gemm[0] != nullptr ||
         gqh_ev_gemm[1] != nullptr || buf[0] != nullptr || buf[1] != nullptr)) {
        const hipError_t err = destroy_resources();
        if (err != hipSuccess) {
            return finish(static_cast<int>(err));
        }
    }
    if (!ready) {
        gqh_resource_device = device_ordinal;
        hipError_t err = hipStreamCreateWithFlags(&gqh_s_dq, hipStreamNonBlocking);
        if (err != hipSuccess) {
            const hipError_t cleanup_err = destroy_resources();
            return finish(cleanup_err != hipSuccess ? static_cast<int>(cleanup_err)
                                                     : static_cast<int>(err));
        }
        err = hipStreamCreateWithFlags(&gqh_s_gemm, hipStreamNonBlocking);
        if (err != hipSuccess) {
            const hipError_t cleanup_err = destroy_resources();
            return finish(cleanup_err != hipSuccess ? static_cast<int>(cleanup_err)
                                                     : static_cast<int>(err));
        }
        for (int i = 0; i < 2; ++i) {
            err = hipEventCreateWithFlags(&ev_dq[i], hipEventDisableTiming);
            if (err != hipSuccess) {
                const hipError_t cleanup_err = destroy_resources();
                return finish(cleanup_err != hipSuccess ? static_cast<int>(cleanup_err)
                                                         : static_cast<int>(err));
            }
            err = hipEventCreateWithFlags(&gqh_ev_gemm[i], hipEventDisableTiming);
            if (err != hipSuccess) {
                const hipError_t cleanup_err = destroy_resources();
                return finish(cleanup_err != hipSuccess ? static_cast<int>(cleanup_err)
                                                         : static_cast<int>(err));
            }
        }
        ready = true;
    }
    const int p = phase & 1;
    GqhPendingWorkGuard pending(device_ordinal);
    const size_t need = static_cast<size_t>(n) * static_cast<size_t>(k);
    if (need > cap[p] || buf[p] == nullptr) {
        if (buf[p] != nullptr) {
            const hipError_t sync_err = hipDeviceSynchronize();
            if (sync_err != hipSuccess) {
                supersonic_gpu_integrity_fail_stop(
                    "GQH dequant buffer reuse synchronize",
                    static_cast<int>(sync_err),
                    device_ordinal);
            }
            const hipError_t free_err = hipFree(buf[p]);
            if (free_err != hipSuccess) {
                supersonic_gpu_integrity_fail_stop(
                    "GQH dequant buffer reuse free",
                    static_cast<int>(free_err),
                    device_ordinal);
            }
            buf[p] = nullptr;
            cap[p] = 0;
        }
        const hipError_t alloc_err = hipMalloc(&buf[p], need * sizeof(hip_bfloat16));
        if (alloc_err != hipSuccess) {
            buf[p] = nullptr;
            return finish(static_cast<int>(alloc_err));
        }
        cap[p] = need;
    }
    // Reuse of buf[p] must wait for the GEMM that last read it.
    if (gqh_ev_gemm_recorded[p]) {
        const hipError_t err = hipEventSynchronize(gqh_ev_gemm[p]);
        if (err != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "GQH GEMM previous event synchronize", static_cast<int>(err), device_ordinal);
        }
    }
    // Decode on the default stream — same launch path as the known-good
    // sequential dequant. Recording the event there, then having s_gemm
    // wait, is the HIP-safe way to overlap decode(N+1) with GEMM(N).
    pending.mark();
    const int dec = supersonic_gqh_hip_decode(
        device_ordinal,
        rung,
        wire,
        tensor_scale,
        grid_code,
        buf[p],
        n,
        k,
        1,
        nullptr);
    if (dec != 0) {
        pending.synchronize_or_fail("GQH dequant decode recovery synchronize");
        return finish(dec);
    }
    {
        const hipError_t err = hipEventRecord(ev_dq[p], nullptr);
        if (err != hipSuccess) {
            pending.synchronize_or_fail("GQH dequant event-record recovery synchronize");
            return finish(static_cast<int>(err));
        }
    }
    {
        const hipError_t err = hipStreamWaitEvent(gqh_s_gemm, ev_dq[p], 0);
        if (err != hipSuccess) {
            pending.synchronize_or_fail("GQH dequant wait recovery synchronize");
            return finish(static_cast<int>(err));
        }
    }
    if (hipblasSetStream(handle, gqh_s_gemm) != HIPBLAS_STATUS_SUCCESS) {
        pending.synchronize_or_fail("GQH dequant stream-set recovery synchronize");
        return finish(7);
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const hipblasStatus_t st = hipblasGemmEx(
        handle,
        HIPBLAS_OP_T,
        HIPBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        buf[p],
        HIP_R_16BF,
        k,
        lhs,
        HIP_R_16BF,
        k,
        &beta,
        out,
        HIP_R_16BF,
        n,
        HIPBLAS_COMPUTE_32F,
        HIPBLAS_GEMM_DEFAULT);
    if (st != HIPBLAS_STATUS_SUCCESS) {
        pending.synchronize_or_fail("GQH GEMM launch recovery synchronize");
        if (hipblasSetStream(handle, nullptr) != HIPBLAS_STATUS_SUCCESS) {
            supersonic_gpu_integrity_fail_stop(
                "GQH GEMM handle stream restore", 8, device_ordinal);
        }
        return finish(8);
    }
    const hipError_t record_err = hipEventRecord(gqh_ev_gemm[p], gqh_s_gemm);
    if (record_err != hipSuccess) {
        pending.synchronize_or_fail("GQH GEMM event-record recovery synchronize");
        if (hipblasSetStream(handle, nullptr) != HIPBLAS_STATUS_SUCCESS) {
            supersonic_gpu_integrity_fail_stop(
                "GQH GEMM handle stream restore", static_cast<int>(record_err), device_ordinal);
        }
        return finish(static_cast<int>(record_err));
    }
    gqh_ev_gemm_recorded[p] = true;
    if (hipblasSetStream(handle, nullptr) != HIPBLAS_STATUS_SUCCESS) {
        pending.synchronize_or_fail("GQH GEMM handle-stream recovery synchronize");
        supersonic_gpu_integrity_fail_stop(
            "GQH GEMM handle stream restore", 10, device_ordinal);
    }
    phase = phase + 1;
    pending.disarm();
    return finish(0);
}

extern "C" int supersonic_qwen35_4b_hip_matmul_rhs_transposed_tiled(
    int dtype,
    size_t device_ordinal,
    size_t batch_elems,
    int m, int n, int k,
    const void* lhs,
    const void* rhs,
    void* out) {
    DecodeBridgeLockGuard guard;
    ScopedHipDevice scoped(static_cast<int>(device_ordinal));
    if (!scoped.ok()) {
        return prefill_backend_failure(268, scoped.status);
    }
    auto finish = [&](int status) {
        const hipError_t restore_err = scoped.restore();
        return status != 0 ? status : static_cast<int>(restore_err);
    };
    switch (dtype) {
    case 2:
        if (matmul_rhs_transposed_hipblas_bf16(
                static_cast<int>(device_ordinal),
                batch_elems,
                m,
                n,
                k,
                lhs,
                rhs,
                out) == 0) {
            return finish(0);
        }
        if (device_supports_wmma_bf16(static_cast<int>(device_ordinal))) {
            const bool disable_small_m = false;
            if (!disable_small_m && m < 32) {
                return finish(matmul_rhs_transposed_wmma_small_m_bf16_device(
                    static_cast<int>(device_ordinal), batch_elems, m, n, k,
                    lhs, rhs, out));
            }
            return finish(matmul_rhs_transposed_tiled_wmma_bf16_device(
                static_cast<int>(device_ordinal), batch_elems, m, n, k,
                lhs, rhs, out));
        }
        return finish(matmul_rhs_transposed_tiled_device<hip_bfloat16>(
            static_cast<int>(device_ordinal), batch_elems, m, n, k,
            lhs, rhs, out));
    default:
        return finish(272);
    }
}

// FP8 dequant matmul for prefill: out = lhs (BF16) × dequant(rhs_fp8)^T
// Uses tiled kernel with 3D grid: (n_tiles, m_tiles, batch)
template <typename T>
int matmul_fp8_dequant_device(
    int device_ordinal,
    size_t batch_elems,
    int m, int n, int k,
    const void* lhs,
    const void* rhs_fp8,
    const void* scale,
    int block_size,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int TILE_M = 16;
    constexpr int TILE_N = 16;
    const int grid_x = (n + TILE_N - 1) / TILE_N;
    const int grid_y = (m + TILE_M - 1) / TILE_M;
    const int grid_z = static_cast<int>(batch_elems);
    const int threads = TILE_M * TILE_N;  // 256
    // Shared memory: s_lhs[16][32] + s_rhs[16][32] = 2 * 16 * 32 * 4 = 4096 bytes
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_matmul_fp8_dequant_kernel<T>),
        dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
        batch_elems, m, n, k,
        static_cast<const T*>(lhs),
        static_cast<const uint8_t*>(rhs_fp8),
        static_cast<const T*>(scale),
        block_size,
        static_cast<T*>(out));
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err = maybe_sync();
    if (launch_err != hipSuccess) return 260;
    if (sync_err != hipSuccess) return 261;
    return 0;
}

// WMMA-tiled FP8 dequant matmul for BF16 activations. Same 64x64 block tile
// shape as the BF16 tiled path; reads FP8 bytes from global, dequantizes
// into LDS as BF16, then runs WMMAs from LDS. Only activated when
// block_size is a multiple of TILED_WMMA_BK=64 so every BK-aligned K slab
// (and the 64-row N slab) lies inside a single FP8 scale block. The
// shipped lovedheart Qwen FP8 bakes use block_size=128.
static int matmul_fp8_dequant_wmma_bf16_device(
    int device_ordinal,
    size_t batch_elems,
    int m, int n, int k,
    const void* lhs,
    const void* rhs_fp8,
    const void* scale,
    int block_size,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;
    const int grid_x = (n + TILE_N - 1) / TILE_N;
    const int grid_y = (m + TILE_M - 1) / TILE_M;
    const int grid_z = static_cast<int>(batch_elems);
    constexpr int threads = 128;  // 4 wavefronts 2x2
    hipLaunchKernelGGL(
        supersonic_qwen35_matmul_fp8_dequant_wmma_kernel,
        dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
        batch_elems, m, n, k,
        static_cast<const hip_bfloat16*>(lhs),
        static_cast<const uint8_t*>(rhs_fp8),
        static_cast<const hip_bfloat16*>(scale),
        block_size,
        static_cast<hip_bfloat16*>(out));
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err = maybe_sync();
    if (launch_err != hipSuccess) return 262;
    if (sync_err != hipSuccess) return 263;
    return 0;
}

extern "C" int supersonic_qwen35_4b_hip_matmul_fp8_dequant(
    int dtype,
    size_t device_ordinal,
    size_t batch_elems,
    int m, int n, int k,
    const void* lhs,
    const void* rhs_fp8,
    const void* scale,
    int block_size,
    void* out) {
    switch (dtype) {
    case 2: {
        // WMMA fast path when block_size is a multiple of the tile's K slab
        // (= 64), and m is large enough that the 64-row tile doesn't waste
        // most of its compute on overhang. Otherwise fall back to the scalar
        // FP32-accumulate tiled kernel (which is what shipped before WMMA).
        constexpr int TILED_BK = 64;
        constexpr int TILED_M_THRESHOLD = 32;
        const int ordinal = static_cast<int>(device_ordinal);
        if (m >= TILED_M_THRESHOLD && block_size % TILED_BK == 0 &&
            device_supports_wmma_bf16(ordinal)) {
            return matmul_fp8_dequant_wmma_bf16_device(
                ordinal, batch_elems, m, n, k,
                lhs, rhs_fp8, scale, block_size, out);
        }
        return matmul_fp8_dequant_device<hip_bfloat16>(
            ordinal, batch_elems, m, n, k,
            lhs, rhs_fp8, scale, block_size, out);
    }
    default:
        return 262;
    }
}

// INT4 dequant matmul bridge.
template <typename T>
int matmul_int4_dequant_device(
    int device_ordinal,
    size_t batch_elems,
    int m, int n, int k,
    const void* lhs,
    const void* rhs_int4,
    const void* scale,
    const void* zero,
    const void* awq_inv_scale,
    int group_size,
    int quant_type,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int TILE_M = 16;
    constexpr int TILE_N = 16;
    const int grid_x = (n + TILE_N - 1) / TILE_N;
    const int grid_y = (m + TILE_M - 1) / TILE_M;
    const int grid_z = static_cast<int>(batch_elems);
    const int threads = TILE_M * TILE_N;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_matmul_int4_dequant_kernel<T>),
        dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
        batch_elems, m, n, k,
        static_cast<const T*>(lhs),
        static_cast<const uint8_t*>(rhs_int4),
        static_cast<const T*>(scale),
        static_cast<const T*>(zero),
        static_cast<const T*>(awq_inv_scale),
        group_size,
        quant_type,
        static_cast<T*>(out));
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err = maybe_sync();
    if (launch_err != hipSuccess) return prefill_backend_failure(270, launch_err);
    if (sync_err != hipSuccess) return prefill_backend_failure(271, sync_err);
    return 0;
}

static int matmul_int4_dequant_wmma_bf16_device(
    int device_ordinal,
    size_t batch_elems,
    int m, int n, int k,
    const void* lhs,
    const void* rhs_int4,
    const void* scale,
    const void* zero,
    const void* awq_inv_scale,
    int group_size,
    int quant_type,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int QWEN35_LOWBIT_NATIVE_INT4 = 4;
    constexpr int QWEN35_LOWBIT_GGML_Q8_0 = 8;
    constexpr int QWEN35_LOWBIT_GGML_Q4_K = 12;
    constexpr int QWEN35_LOWBIT_GGML_Q5_K = 13;
    constexpr int QWEN35_LOWBIT_GGML_Q6_K = 14;
    const bool native_int4 = quant_type == QWEN35_LOWBIT_NATIVE_INT4;
    const bool ggml_k =
        quant_type == QWEN35_LOWBIT_GGML_Q8_0 ||
        quant_type == QWEN35_LOWBIT_GGML_Q4_K ||
        quant_type == QWEN35_LOWBIT_GGML_Q5_K ||
        quant_type == QWEN35_LOWBIT_GGML_Q6_K;
    const bool disable_ggml_small_m = false;

    // Tiled WMMA is a clear win when M is large enough to use most of the
    // 64-row block tile (long-ctx prefill). For small M (short prompts,
    // decode-like verify blocks) the 4x row overhang dominates, so dispatch
    // to the one-wave-per-16x16-tile kernel in that regime, including raw
    // GGML K-block vocab projections.
    constexpr int TILED_M_THRESHOLD = 32;
    const bool raw_ggml_small_m =
        ggml_k && !disable_ggml_small_m;
    if ((native_int4 || raw_ggml_small_m) && m < TILED_M_THRESHOLD) {
        const bool enable_ggml_small_m_n64 = false;
        if (raw_ggml_small_m && enable_ggml_small_m_n64) {
            constexpr int TILE_M = 16;
            constexpr int TILE_N = 64;
            const int grid_x = (n + TILE_N - 1) / TILE_N;
            const int grid_y = (m + TILE_M - 1) / TILE_M;
            const int grid_z = static_cast<int>(batch_elems);
            const int threads = 128;
            hipLaunchKernelGGL(
                supersonic_qwen35_matmul_int4_dequant_wmma_small_m_n64_kernel,
                dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
                batch_elems, m, n, k,
                static_cast<const hip_bfloat16*>(lhs),
                static_cast<const uint8_t*>(rhs_int4),
                static_cast<const hip_bfloat16*>(scale),
                static_cast<const hip_bfloat16*>(zero),
                static_cast<const hip_bfloat16*>(awq_inv_scale),
                group_size,
                quant_type,
                static_cast<hip_bfloat16*>(out));
            hipError_t launch_err = hipGetLastError();
            hipError_t sync_err = maybe_sync();
            if (launch_err != hipSuccess) return prefill_backend_failure(292, launch_err);
            if (sync_err != hipSuccess) return prefill_backend_failure(293, sync_err);
            return 0;
        }

        const bool disable_ggml_small_m_qtype = false;
        if (raw_ggml_small_m && !disable_ggml_small_m_qtype) {
            const bool trunc_dequant = quant_type != QWEN35_LOWBIT_GGML_Q8_0;
            constexpr int TILE_M = 16;
            constexpr int TILE_N = 16;
            const int grid_x = (n + TILE_N - 1) / TILE_N;
            const int grid_y = (m + TILE_M - 1) / TILE_M;
            const int grid_z = static_cast<int>(batch_elems);
            const int threads = 32;
            const bool use_gfx12_acc = device_is_gfx12(device_ordinal);
            const bool enable_m8_block =
                m == 8 && (n % TILE_N) == 0 && (k % 256) == 0 && awq_inv_scale == nullptr;
            const bool enable_m16_block =
                m == 16 && (n % TILE_N) == 0 && (k % 256) == 0 && awq_inv_scale == nullptr;
            if (enable_m16_block) {
                if (quant_type == QWEN35_LOWBIT_GGML_Q8_0) {
                    if (trunc_dequant) {
                        if (use_gfx12_acc) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q8_0, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q8_0, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        }
                    } else {
                        if (use_gfx12_acc) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q8_0, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q8_0, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        }
                    }
                } else if (quant_type == QWEN35_LOWBIT_GGML_Q4_K) {
                    if (trunc_dequant) {
                        if (use_gfx12_acc) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q4_K, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q4_K, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        }
                    } else {
                        if (use_gfx12_acc) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q4_K, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q4_K, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        }
                    }
                } else if (quant_type == QWEN35_LOWBIT_GGML_Q5_K) {
                    if (trunc_dequant) {
                        if (use_gfx12_acc) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q5_K, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q5_K, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        }
                    } else {
                        if (use_gfx12_acc) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q5_K, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q5_K, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        }
                    }
                } else if (quant_type == QWEN35_LOWBIT_GGML_Q6_K) {
                    if (trunc_dequant) {
                        if (use_gfx12_acc) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q6_K, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q6_K, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        }
                    } else {
                        if (use_gfx12_acc) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q6_K, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q6_K, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                nullptr,
                                static_cast<hip_bfloat16*>(out));
                        }
                    }
                }
                hipError_t launch_err = hipGetLastError();
                hipError_t sync_err = maybe_sync();
                if (launch_err != hipSuccess) return prefill_backend_failure(298, launch_err);
                if (sync_err != hipSuccess) return prefill_backend_failure(299, sync_err);
                return 0;
            }
            if (enable_m8_block) {
                const bool use_m8_gfx12 = use_gfx12_acc;
                if (quant_type == QWEN35_LOWBIT_GGML_Q8_0) {
                    if (trunc_dequant) {
                        if (use_m8_gfx12) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q8_0, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_kernel<QWEN35_LOWBIT_GGML_Q8_0, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        }
                    } else {
                        if (use_m8_gfx12) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q8_0, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_kernel<QWEN35_LOWBIT_GGML_Q8_0, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        }
                    }
                } else if (quant_type == QWEN35_LOWBIT_GGML_Q4_K) {
                    if (trunc_dequant) {
                        if (use_m8_gfx12) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q4_K, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_kernel<QWEN35_LOWBIT_GGML_Q4_K, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        }
                    } else {
                        if (use_m8_gfx12) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q4_K, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_kernel<QWEN35_LOWBIT_GGML_Q4_K, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        }
                    }
                } else if (quant_type == QWEN35_LOWBIT_GGML_Q5_K) {
                    if (trunc_dequant) {
                        if (use_m8_gfx12) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q5_K, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_kernel<QWEN35_LOWBIT_GGML_Q5_K, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        }
                    } else {
                        if (use_m8_gfx12) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q5_K, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_kernel<QWEN35_LOWBIT_GGML_Q5_K, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        }
                    }
                } else if (quant_type == QWEN35_LOWBIT_GGML_Q6_K) {
                    if (trunc_dequant) {
                        if (use_m8_gfx12) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q6_K, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_kernel<QWEN35_LOWBIT_GGML_Q6_K, true>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        }
                    } else {
                        if (use_m8_gfx12) {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q6_K, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        } else {
                            hipLaunchKernelGGL(
                                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m8_qtype_kernel<QWEN35_LOWBIT_GGML_Q6_K, false>),
                                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                                batch_elems, n, k,
                                static_cast<const hip_bfloat16*>(lhs),
                                static_cast<const uint8_t*>(rhs_int4),
                                static_cast<hip_bfloat16*>(out));
                        }
                    }
                }
                hipError_t launch_err = hipGetLastError();
                hipError_t sync_err = maybe_sync();
                if (launch_err != hipSuccess) return prefill_backend_failure(296, launch_err);
                if (sync_err != hipSuccess) return prefill_backend_failure(297, sync_err);
                return 0;
            }
            if (quant_type == QWEN35_LOWBIT_GGML_Q8_0) {
                if (trunc_dequant) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_small_m_qtype_kernel<QWEN35_LOWBIT_GGML_Q8_0, true>),
                        dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
                        batch_elems, m, n, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_int4),
                        static_cast<const hip_bfloat16*>(awq_inv_scale),
                        static_cast<hip_bfloat16*>(out));
                } else {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_small_m_qtype_kernel<QWEN35_LOWBIT_GGML_Q8_0, false>),
                        dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
                        batch_elems, m, n, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_int4),
                        static_cast<const hip_bfloat16*>(awq_inv_scale),
                        static_cast<hip_bfloat16*>(out));
                }
            } else if (quant_type == QWEN35_LOWBIT_GGML_Q4_K) {
                if (trunc_dequant) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_small_m_qtype_kernel<QWEN35_LOWBIT_GGML_Q4_K, true>),
                        dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
                        batch_elems, m, n, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_int4),
                        static_cast<const hip_bfloat16*>(awq_inv_scale),
                        static_cast<hip_bfloat16*>(out));
                } else {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_small_m_qtype_kernel<QWEN35_LOWBIT_GGML_Q4_K, false>),
                        dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
                        batch_elems, m, n, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_int4),
                        static_cast<const hip_bfloat16*>(awq_inv_scale),
                        static_cast<hip_bfloat16*>(out));
                }
            } else if (quant_type == QWEN35_LOWBIT_GGML_Q5_K) {
                if (trunc_dequant) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_small_m_qtype_kernel<QWEN35_LOWBIT_GGML_Q5_K, true>),
                        dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
                        batch_elems, m, n, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_int4),
                        static_cast<const hip_bfloat16*>(awq_inv_scale),
                        static_cast<hip_bfloat16*>(out));
                } else {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_small_m_qtype_kernel<QWEN35_LOWBIT_GGML_Q5_K, false>),
                        dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
                        batch_elems, m, n, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_int4),
                        static_cast<const hip_bfloat16*>(awq_inv_scale),
                        static_cast<hip_bfloat16*>(out));
                }
            } else if (quant_type == QWEN35_LOWBIT_GGML_Q6_K) {
                if (trunc_dequant) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_small_m_qtype_kernel<QWEN35_LOWBIT_GGML_Q6_K, true>),
                        dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
                        batch_elems, m, n, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_int4),
                        static_cast<const hip_bfloat16*>(awq_inv_scale),
                        static_cast<hip_bfloat16*>(out));
                } else {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_small_m_qtype_kernel<QWEN35_LOWBIT_GGML_Q6_K, false>),
                        dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
                        batch_elems, m, n, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_int4),
                        static_cast<const hip_bfloat16*>(awq_inv_scale),
                        static_cast<hip_bfloat16*>(out));
                }
            }
            hipError_t launch_err = hipGetLastError();
            hipError_t sync_err = maybe_sync();
            if (launch_err != hipSuccess) return prefill_backend_failure(294, launch_err);
            if (sync_err != hipSuccess) return prefill_backend_failure(295, sync_err);
            return 0;
        }

        constexpr int TILE_M = 16;
        constexpr int TILE_N = 16;
        const int grid_x = (n + TILE_N - 1) / TILE_N;
        const int grid_y = (m + TILE_M - 1) / TILE_M;
        const int grid_z = static_cast<int>(batch_elems);
        const int threads = 32;
        hipLaunchKernelGGL(
            supersonic_qwen35_matmul_int4_dequant_wmma_small_m_kernel,
            dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
            batch_elems, m, n, k,
            static_cast<const hip_bfloat16*>(lhs),
            static_cast<const uint8_t*>(rhs_int4),
            static_cast<const hip_bfloat16*>(scale),
            static_cast<const hip_bfloat16*>(zero),
            static_cast<const hip_bfloat16*>(awq_inv_scale),
            group_size,
            quant_type,
            static_cast<hip_bfloat16*>(out));
        hipError_t launch_err = hipGetLastError();
        hipError_t sync_err = maybe_sync();
        if (launch_err != hipSuccess) return prefill_backend_failure(290, launch_err);
        if (sync_err != hipSuccess) return prefill_backend_failure(291, sync_err);
        return 0;
    }

    // Large-M: tiled 64x64 block tile, 4 waves in 2x2.
    constexpr int TILE_M = 64;
    constexpr int TILE_N = 64;
    const int grid_x = (n + TILE_N - 1) / TILE_N;
    const int grid_y = (m + TILE_M - 1) / TILE_M;
    const int grid_z = static_cast<int>(batch_elems);
    const int threads = 128;
    const bool use_gfx12_large_q8 =
        device_is_gfx12(device_ordinal) &&
        quant_type == QWEN35_LOWBIT_GGML_Q8_0 &&
        awq_inv_scale == nullptr &&
        (n % TILE_N) == 0 &&
        (k % 64) == 0;
    if (use_gfx12_large_q8) {
        hipLaunchKernelGGL(
            supersonic_qwen35_matmul_ggml_q8_0_wmma_gfx12_large_m_kernel,
            dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
            batch_elems, m, n, k,
            static_cast<const hip_bfloat16*>(lhs),
            static_cast<const uint8_t*>(rhs_int4),
            static_cast<hip_bfloat16*>(out));
        hipError_t launch_err = hipGetLastError();
        hipError_t sync_err = maybe_sync();
        if (launch_err != hipSuccess) return prefill_backend_failure(300, launch_err);
        if (sync_err != hipSuccess) return prefill_backend_failure(301, sync_err);
        return 0;
    }
    hipLaunchKernelGGL(
        supersonic_qwen35_matmul_int4_dequant_wmma_kernel,
        dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
        batch_elems, m, n, k,
        static_cast<const hip_bfloat16*>(lhs),
        static_cast<const uint8_t*>(rhs_int4),
        static_cast<const hip_bfloat16*>(scale),
        static_cast<const hip_bfloat16*>(zero),
        static_cast<const hip_bfloat16*>(awq_inv_scale),
        group_size,
        quant_type,
        static_cast<hip_bfloat16*>(out));
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err = maybe_sync();
    if (launch_err != hipSuccess) return prefill_backend_failure(290, launch_err);
    if (sync_err != hipSuccess) return prefill_backend_failure(291, sync_err);
    return 0;
}

static int matmul_int4_dequant_residual_add_bf16_device(
    int device_ordinal,
    size_t batch_elems,
    int m, int n, int k,
    const void* lhs,
    const void* rhs_int4,
    const void* awq_inv_scale,
    int quant_type,
    const void* residual,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int QWEN35_LOWBIT_GGML_Q8_0 = 8;
    constexpr int QWEN35_LOWBIT_GGML_Q4_K = 12;
    constexpr int QWEN35_LOWBIT_GGML_Q5_K = 13;
    constexpr int QWEN35_LOWBIT_GGML_Q6_K = 14;
    const bool ggml_k =
        quant_type == QWEN35_LOWBIT_GGML_Q8_0 ||
        quant_type == QWEN35_LOWBIT_GGML_Q4_K ||
        quant_type == QWEN35_LOWBIT_GGML_Q5_K ||
        quant_type == QWEN35_LOWBIT_GGML_Q6_K;
    if (!ggml_k) return 312;
    if (!device_supports_wmma_bf16(device_ordinal)) return 313;
    if (lhs == nullptr || rhs_int4 == nullptr || residual == nullptr || out == nullptr) return 314;
    if (m != 16 || n <= 0 || k <= 0 || (n % 16) != 0 || (k % 256) != 0) return 315;
    if (awq_inv_scale != nullptr) return 316;

    const bool trunc_dequant = quant_type != QWEN35_LOWBIT_GGML_Q8_0;
    constexpr int TILE_N = 16;
    const int grid_x = (n + TILE_N - 1) / TILE_N;
    const int grid_z = static_cast<int>(batch_elems);
    const int threads = 32;
    const bool use_gfx12_acc = device_is_gfx12(device_ordinal);

    if (quant_type == QWEN35_LOWBIT_GGML_Q8_0) {
        if (trunc_dequant) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q8_0, true, true>),
                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                batch_elems, n, k,
                static_cast<const hip_bfloat16*>(lhs),
                static_cast<const uint8_t*>(rhs_int4),
                static_cast<const hip_bfloat16*>(residual),
                static_cast<hip_bfloat16*>(out));
        } else {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q8_0, false, true>),
                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                batch_elems, n, k,
                static_cast<const hip_bfloat16*>(lhs),
                static_cast<const uint8_t*>(rhs_int4),
                static_cast<const hip_bfloat16*>(residual),
                static_cast<hip_bfloat16*>(out));
        }
    } else if (quant_type == QWEN35_LOWBIT_GGML_Q4_K) {
        if (trunc_dequant) {
            if (use_gfx12_acc) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q4_K, true, true>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_int4),
                    static_cast<const hip_bfloat16*>(residual),
                    static_cast<hip_bfloat16*>(out));
            } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q4_K, true, true>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_int4),
                    static_cast<const hip_bfloat16*>(residual),
                    static_cast<hip_bfloat16*>(out));
            }
        } else {
            if (use_gfx12_acc) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q4_K, false, true>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_int4),
                    static_cast<const hip_bfloat16*>(residual),
                    static_cast<hip_bfloat16*>(out));
            } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q4_K, false, true>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_int4),
                    static_cast<const hip_bfloat16*>(residual),
                    static_cast<hip_bfloat16*>(out));
            }
        }
    } else if (quant_type == QWEN35_LOWBIT_GGML_Q5_K) {
        if (trunc_dequant) {
            if (use_gfx12_acc) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q5_K, true, true>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_int4),
                    static_cast<const hip_bfloat16*>(residual),
                    static_cast<hip_bfloat16*>(out));
            } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q5_K, true, true>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_int4),
                    static_cast<const hip_bfloat16*>(residual),
                    static_cast<hip_bfloat16*>(out));
            }
        } else {
            if (use_gfx12_acc) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q5_K, false, true>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_int4),
                    static_cast<const hip_bfloat16*>(residual),
                    static_cast<hip_bfloat16*>(out));
            } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q5_K, false, true>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_int4),
                    static_cast<const hip_bfloat16*>(residual),
                    static_cast<hip_bfloat16*>(out));
            }
        }
    } else if (quant_type == QWEN35_LOWBIT_GGML_Q6_K) {
        if (trunc_dequant) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q6_K, true, true>),
                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                batch_elems, n, k,
                static_cast<const hip_bfloat16*>(lhs),
                static_cast<const uint8_t*>(rhs_int4),
                static_cast<const hip_bfloat16*>(residual),
                static_cast<hip_bfloat16*>(out));
        } else {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_dequant_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q6_K, false, true>),
                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                batch_elems, n, k,
                static_cast<const hip_bfloat16*>(lhs),
                static_cast<const uint8_t*>(rhs_int4),
                static_cast<const hip_bfloat16*>(residual),
                static_cast<hip_bfloat16*>(out));
        }
    }

    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err = maybe_sync();
    if (launch_err != hipSuccess) return 317;
    if (sync_err != hipSuccess) return 318;
    return 0;
}

static int matmul_ggml_pair_wmma_bf16_device(
    int device_ordinal,
    size_t batch_elems,
    int m, int n_each, int k,
    const void* lhs,
    const void* rhs_first,
    const void* rhs_second,
    int quant_type,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int QWEN35_LOWBIT_GGML_Q8_0 = 8;
    constexpr int QWEN35_LOWBIT_GGML_Q4_K = 12;
    constexpr int QWEN35_LOWBIT_GGML_Q5_K = 13;
    constexpr int QWEN35_LOWBIT_GGML_Q6_K = 14;
    const bool ggml_k =
        quant_type == QWEN35_LOWBIT_GGML_Q8_0 ||
        quant_type == QWEN35_LOWBIT_GGML_Q4_K ||
        quant_type == QWEN35_LOWBIT_GGML_Q5_K ||
        quant_type == QWEN35_LOWBIT_GGML_Q6_K;
    if (!ggml_k) return 294;
    if (!device_supports_wmma_bf16(device_ordinal)) return 295;

    constexpr int TILE_M = 16;
    constexpr int TILE_N = 16;
    const bool enable_m16_qtype =
        m == TILE_M && (n_each % TILE_N) == 0 && (k % 256) == 0;
    if (enable_m16_qtype) {
        const int n_out = n_each * 2;
        const int grid_x = (n_out + TILE_N - 1) / TILE_N;
        const int grid_z = static_cast<int>(batch_elems);
        const int threads = 32;
        const bool use_gfx12_acc = device_is_gfx12(device_ordinal);
        const bool trunc_dequant = quant_type != QWEN35_LOWBIT_GGML_Q8_0;
        if (quant_type == QWEN35_LOWBIT_GGML_Q8_0) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q8_0, false>),
                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                batch_elems, n_each, k,
                static_cast<const hip_bfloat16*>(lhs),
                static_cast<const uint8_t*>(rhs_first),
                static_cast<const uint8_t*>(rhs_second),
                static_cast<hip_bfloat16*>(out));
        } else if (quant_type == QWEN35_LOWBIT_GGML_Q4_K) {
            if (trunc_dequant) {
                if (use_gfx12_acc) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q4_K, true>),
                        dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                        batch_elems, n_each, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_first),
                        static_cast<const uint8_t*>(rhs_second),
                        static_cast<hip_bfloat16*>(out));
                } else {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q4_K, true>),
                        dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                        batch_elems, n_each, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_first),
                        static_cast<const uint8_t*>(rhs_second),
                        static_cast<hip_bfloat16*>(out));
                }
            } else {
                if (use_gfx12_acc) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q4_K, false>),
                        dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                        batch_elems, n_each, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_first),
                        static_cast<const uint8_t*>(rhs_second),
                        static_cast<hip_bfloat16*>(out));
                } else {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q4_K, false>),
                        dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                        batch_elems, n_each, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_first),
                        static_cast<const uint8_t*>(rhs_second),
                        static_cast<hip_bfloat16*>(out));
                }
            }
        } else if (quant_type == QWEN35_LOWBIT_GGML_Q5_K) {
            if (trunc_dequant) {
                if (use_gfx12_acc) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q5_K, true>),
                        dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                        batch_elems, n_each, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_first),
                        static_cast<const uint8_t*>(rhs_second),
                        static_cast<hip_bfloat16*>(out));
                } else {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q5_K, true>),
                        dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                        batch_elems, n_each, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_first),
                        static_cast<const uint8_t*>(rhs_second),
                        static_cast<hip_bfloat16*>(out));
                }
            } else {
                if (use_gfx12_acc) {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q5_K, false>),
                        dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                        batch_elems, n_each, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_first),
                        static_cast<const uint8_t*>(rhs_second),
                        static_cast<hip_bfloat16*>(out));
                } else {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q5_K, false>),
                        dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                        batch_elems, n_each, k,
                        static_cast<const hip_bfloat16*>(lhs),
                        static_cast<const uint8_t*>(rhs_first),
                        static_cast<const uint8_t*>(rhs_second),
                        static_cast<hip_bfloat16*>(out));
                }
            }
        } else if (quant_type == QWEN35_LOWBIT_GGML_Q6_K) {
            if (trunc_dequant) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q6_K, true>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n_each, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_first),
                    static_cast<const uint8_t*>(rhs_second),
                    static_cast<hip_bfloat16*>(out));
            } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q6_K, false>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n_each, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_first),
                    static_cast<const uint8_t*>(rhs_second),
                    static_cast<hip_bfloat16*>(out));
            }
        }
        hipError_t launch_err = hipGetLastError();
        hipError_t sync_err = maybe_sync();
        if (launch_err != hipSuccess) return 320;
        if (sync_err != hipSuccess) return 321;
        return 0;
    }

    const int n_out = n_each * 2;
    const int grid_x = (n_out + TILE_N - 1) / TILE_N;
    const int grid_y = (m + TILE_M - 1) / TILE_M;
    const int grid_z = static_cast<int>(batch_elems);
    const int threads = 32;
    hipLaunchKernelGGL(
        supersonic_qwen35_matmul_ggml_pair_wmma_small_m_kernel,
        dim3(grid_x, grid_y, grid_z), dim3(threads), 0, 0,
        batch_elems, m, n_each, k,
        static_cast<const hip_bfloat16*>(lhs),
        static_cast<const uint8_t*>(rhs_first),
        static_cast<const uint8_t*>(rhs_second),
        quant_type,
        static_cast<hip_bfloat16*>(out));
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err = maybe_sync();
    if (launch_err != hipSuccess) return 296;
    if (sync_err != hipSuccess) return 297;
    return 0;
}

static int matmul_ggml_pair_swiglu_wmma_bf16_device(
    int device_ordinal,
    size_t batch_elems,
    int m, int n_each, int k,
    const void* lhs,
    const void* rhs_gate,
    const void* rhs_up,
    int quant_type,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int QWEN35_LOWBIT_GGML_Q8_0 = 8;
    constexpr int QWEN35_LOWBIT_GGML_Q4_K = 12;
    constexpr int QWEN35_LOWBIT_GGML_Q5_K = 13;
    constexpr int QWEN35_LOWBIT_GGML_Q6_K = 14;
    const bool ggml_k =
        quant_type == QWEN35_LOWBIT_GGML_Q8_0 ||
        quant_type == QWEN35_LOWBIT_GGML_Q4_K ||
        quant_type == QWEN35_LOWBIT_GGML_Q5_K ||
        quant_type == QWEN35_LOWBIT_GGML_Q6_K;
    if (!ggml_k) return 322;
    if (!device_supports_wmma_bf16(device_ordinal)) return 323;

    constexpr int TILE_M = 16;
    constexpr int TILE_N = 16;
    if (m != TILE_M || (n_each % TILE_N) != 0 || (k % 256) != 0) return 324;

    const int grid_x = (n_each + TILE_N - 1) / TILE_N;
    const int grid_z = static_cast<int>(batch_elems);
    const int threads = 32;
    const bool trunc_dequant = quant_type != QWEN35_LOWBIT_GGML_Q8_0;
    const bool use_gfx12_acc = device_is_gfx12(device_ordinal);
    if (quant_type == QWEN35_LOWBIT_GGML_Q8_0) {
        if (use_gfx12_acc) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_swiglu_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q8_0, false>),
                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                batch_elems, n_each, k,
                static_cast<const hip_bfloat16*>(lhs),
                static_cast<const uint8_t*>(rhs_gate),
                static_cast<const uint8_t*>(rhs_up),
                static_cast<hip_bfloat16*>(out));
        } else {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_swiglu_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q8_0, false>),
                dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                batch_elems, n_each, k,
                static_cast<const hip_bfloat16*>(lhs),
                static_cast<const uint8_t*>(rhs_gate),
                static_cast<const uint8_t*>(rhs_up),
                static_cast<hip_bfloat16*>(out));
        }
    } else if (quant_type == QWEN35_LOWBIT_GGML_Q4_K) {
        if (trunc_dequant) {
            if (use_gfx12_acc) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_swiglu_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q4_K, true>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n_each, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_gate),
                    static_cast<const uint8_t*>(rhs_up),
                    static_cast<hip_bfloat16*>(out));
            } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_swiglu_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q4_K, true>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n_each, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_gate),
                    static_cast<const uint8_t*>(rhs_up),
                    static_cast<hip_bfloat16*>(out));
            }
        } else {
            if (use_gfx12_acc) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_swiglu_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q4_K, false>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n_each, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_gate),
                    static_cast<const uint8_t*>(rhs_up),
                    static_cast<hip_bfloat16*>(out));
            } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_swiglu_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q4_K, false>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n_each, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_gate),
                    static_cast<const uint8_t*>(rhs_up),
                    static_cast<hip_bfloat16*>(out));
            }
        }
    } else if (quant_type == QWEN35_LOWBIT_GGML_Q5_K) {
        if (trunc_dequant) {
            if (use_gfx12_acc) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_swiglu_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q5_K, true>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n_each, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_gate),
                    static_cast<const uint8_t*>(rhs_up),
                    static_cast<hip_bfloat16*>(out));
            } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_swiglu_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q5_K, true>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n_each, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_gate),
                    static_cast<const uint8_t*>(rhs_up),
                    static_cast<hip_bfloat16*>(out));
            }
        } else {
            if (use_gfx12_acc) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_swiglu_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q5_K, false>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n_each, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_gate),
                    static_cast<const uint8_t*>(rhs_up),
                    static_cast<hip_bfloat16*>(out));
            } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_swiglu_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q5_K, false>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n_each, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_gate),
                    static_cast<const uint8_t*>(rhs_up),
                    static_cast<hip_bfloat16*>(out));
            }
        }
    } else if (quant_type == QWEN35_LOWBIT_GGML_Q6_K) {
        if (trunc_dequant) {
            if (use_gfx12_acc) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_swiglu_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q6_K, true>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n_each, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_gate),
                    static_cast<const uint8_t*>(rhs_up),
                    static_cast<hip_bfloat16*>(out));
            } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_swiglu_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q6_K, true>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n_each, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_gate),
                    static_cast<const uint8_t*>(rhs_up),
                    static_cast<hip_bfloat16*>(out));
            }
        } else {
            if (use_gfx12_acc) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_swiglu_wmma_m16_qtype_gfx12_kernel<QWEN35_LOWBIT_GGML_Q6_K, false>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n_each, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_gate),
                    static_cast<const uint8_t*>(rhs_up),
                    static_cast<hip_bfloat16*>(out));
            } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_matmul_ggml_pair_swiglu_wmma_m16_qtype_kernel<QWEN35_LOWBIT_GGML_Q6_K, false>),
                    dim3(grid_x, 1, grid_z), dim3(threads), 0, 0,
                    batch_elems, n_each, k,
                    static_cast<const hip_bfloat16*>(lhs),
                    static_cast<const uint8_t*>(rhs_gate),
                    static_cast<const uint8_t*>(rhs_up),
                    static_cast<hip_bfloat16*>(out));
            }
        }
    }
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err = maybe_sync();
    if (launch_err != hipSuccess) return 325;
    if (sync_err != hipSuccess) return 326;
    return 0;
}

template <typename T>
static int quantize_mmq_q8_1_device(
    int device_ordinal,
    size_t batch_elems,
    int m,
    int k,
    const void* lhs,
    int quant_type,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int QWEN35_LOWBIT_GGML_Q4_K = 12;
    constexpr int QWEN35_LOWBIT_GGML_Q5_K = 13;
    constexpr int QWEN35_LOWBIT_GGML_Q6_K = 14;
    const bool ggml_k =
        quant_type == QWEN35_LOWBIT_GGML_Q4_K ||
        quant_type == QWEN35_LOWBIT_GGML_Q5_K ||
        quant_type == QWEN35_LOWBIT_GGML_Q6_K;
    if (!ggml_k) return 300;
    if (m <= 0 || k <= 0 || (k % 128) != 0) return 301;

    const int blocks_per_row = (k + 127) / 128;
    hipLaunchKernelGGL(
        supersonic_qwen35_quantize_mmq_q8_1_kernel<T>,
        dim3(blocks_per_row, m, static_cast<unsigned int>(batch_elems)),
        dim3(32),
        0,
        0,
        batch_elems,
        m,
        k,
        static_cast<const T*>(lhs),
        quant_type,
        static_cast<uint8_t*>(out));
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err = maybe_sync();
    if (launch_err != hipSuccess) return 302;
    if (sync_err != hipSuccess) return 303;
    return 0;
}

static int matmul_mmq_q8_1_q6_k_device(
    int device_ordinal,
    size_t batch_elems,
    int m,
    int n,
    int k,
    const void* q8,
    const void* rhs_q6,
    const void* residual,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    if (!device_supports_wmma_i8(device_ordinal)) return 309;
    if (m <= 0 || n <= 0 || k <= 0 || (k % 256) != 0) return 305;
    constexpr int MMQ_X = 16;
    constexpr int MMQ_Y = 128;
    constexpr int PADDED_TILE_Y_INTS = 768;
    constexpr int MMQ_MMA_TILE_X_K_Q6_K = 76;
    const int grid_x = (n + MMQ_Y - 1) / MMQ_Y;
    const int grid_y = (m + MMQ_X - 1) / MMQ_X;
    const int grid_z = static_cast<int>(batch_elems);
    const size_t shared_bytes =
        static_cast<size_t>(PADDED_TILE_Y_INTS + MMQ_Y * MMQ_MMA_TILE_X_K_Q6_K) *
        sizeof(int);
    const bool hot_exact = m == MMQ_X && (n % MMQ_Y) == 0;
    const bool use_gfx12_native =
        hot_exact && device_is_gfx12(device_ordinal);
    const bool has_residual = residual != nullptr;

    if (use_gfx12_native && has_residual) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(supersonic_qwen35_matmul_mmq_q8_1_q6_k_gfx12_kernel<true, true>),
            dim3(grid_x, grid_y, grid_z),
            dim3(32, 8),
            shared_bytes,
            0,
            batch_elems,
            m,
            n,
            k,
            static_cast<const uint8_t*>(q8),
            static_cast<const uint8_t*>(rhs_q6),
            static_cast<const hip_bfloat16*>(residual),
            static_cast<hip_bfloat16*>(out));
    } else if (use_gfx12_native) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(supersonic_qwen35_matmul_mmq_q8_1_q6_k_gfx12_kernel<true, false>),
            dim3(grid_x, grid_y, grid_z),
            dim3(32, 8),
            shared_bytes,
            0,
            batch_elems,
            m,
            n,
            k,
            static_cast<const uint8_t*>(q8),
            static_cast<const uint8_t*>(rhs_q6),
            static_cast<const hip_bfloat16*>(residual),
            static_cast<hip_bfloat16*>(out));
    } else if (hot_exact && has_residual) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(supersonic_qwen35_matmul_mmq_q8_1_q6_k_kernel<true, true>),
            dim3(grid_x, grid_y, grid_z),
            dim3(32, 8),
            shared_bytes,
            0,
            batch_elems,
            m,
            n,
            k,
            static_cast<const uint8_t*>(q8),
            static_cast<const uint8_t*>(rhs_q6),
            static_cast<const hip_bfloat16*>(residual),
            static_cast<hip_bfloat16*>(out));
    } else if (hot_exact) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(supersonic_qwen35_matmul_mmq_q8_1_q6_k_kernel<true, false>),
            dim3(grid_x, grid_y, grid_z),
            dim3(32, 8),
            shared_bytes,
            0,
            batch_elems,
            m,
            n,
            k,
            static_cast<const uint8_t*>(q8),
            static_cast<const uint8_t*>(rhs_q6),
            static_cast<const hip_bfloat16*>(residual),
            static_cast<hip_bfloat16*>(out));
    } else if (has_residual) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(supersonic_qwen35_matmul_mmq_q8_1_q6_k_kernel<false, true>),
            dim3(grid_x, grid_y, grid_z),
            dim3(32, 8),
            shared_bytes,
            0,
            batch_elems,
            m,
            n,
            k,
            static_cast<const uint8_t*>(q8),
            static_cast<const uint8_t*>(rhs_q6),
            static_cast<const hip_bfloat16*>(residual),
            static_cast<hip_bfloat16*>(out));
    } else {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(supersonic_qwen35_matmul_mmq_q8_1_q6_k_kernel<false, false>),
            dim3(grid_x, grid_y, grid_z),
            dim3(32, 8),
            shared_bytes,
            0,
            batch_elems,
            m,
            n,
            k,
            static_cast<const uint8_t*>(q8),
            static_cast<const uint8_t*>(rhs_q6),
            static_cast<const hip_bfloat16*>(residual),
            static_cast<hip_bfloat16*>(out));
    }
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err = maybe_sync();
    if (launch_err != hipSuccess) return 306;
    if (sync_err != hipSuccess) return 307;
    return 0;
}

static int matmul_q6_k_m16_argmax_device(
    int device_ordinal,
    size_t batch_elems,
    int m,
    int n,
    int k,
    const void* lhs,
    const void* rhs_q6,
    void* block_best_vals,
    void* block_best_indices,
    void* out_indices
) {
    ScopedHipDevice scoped(device_ordinal);
    if (!device_supports_wmma_bf16(device_ordinal)) return 340;
    if (m != 16 || n <= 0 || k <= 0 || (n % 16) != 0 || (k % 256) != 0) return 341;
    if (lhs == nullptr || rhs_q6 == nullptr || block_best_vals == nullptr ||
        block_best_indices == nullptr || out_indices == nullptr) return 342;

    const int tiles = n / 16;
    const int grid_z = static_cast<int>(batch_elems);
    const bool trunc_dequant = true;
    const bool use_gfx12_acc = device_is_gfx12(device_ordinal);
    if (trunc_dequant) {
        if (use_gfx12_acc) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(supersonic_qwen35_matmul_q6_k_m16_tile_argmax_gfx12_kernel<true>),
                dim3(tiles, 1, grid_z),
                dim3(32),
                0,
                0,
                batch_elems,
                n,
                k,
                static_cast<const hip_bfloat16*>(lhs),
                static_cast<const uint8_t*>(rhs_q6),
                static_cast<float*>(block_best_vals),
                static_cast<uint32_t*>(block_best_indices));
        } else {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(supersonic_qwen35_matmul_q6_k_m16_tile_argmax_kernel<true>),
                dim3(tiles, 1, grid_z),
                dim3(32),
                0,
                0,
                batch_elems,
                n,
                k,
                static_cast<const hip_bfloat16*>(lhs),
                static_cast<const uint8_t*>(rhs_q6),
                static_cast<float*>(block_best_vals),
                static_cast<uint32_t*>(block_best_indices));
        }
    } else {
        if (use_gfx12_acc) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(supersonic_qwen35_matmul_q6_k_m16_tile_argmax_gfx12_kernel<false>),
                dim3(tiles, 1, grid_z),
                dim3(32),
                0,
                0,
                batch_elems,
                n,
                k,
                static_cast<const hip_bfloat16*>(lhs),
                static_cast<const uint8_t*>(rhs_q6),
                static_cast<float*>(block_best_vals),
                static_cast<uint32_t*>(block_best_indices));
        } else {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(supersonic_qwen35_matmul_q6_k_m16_tile_argmax_kernel<false>),
                dim3(tiles, 1, grid_z),
                dim3(32),
                0,
                0,
                batch_elems,
                n,
                k,
                static_cast<const hip_bfloat16*>(lhs),
                static_cast<const uint8_t*>(rhs_q6),
                static_cast<float*>(block_best_vals),
                static_cast<uint32_t*>(block_best_indices));
        }
    }
    hipError_t launch_err = hipGetLastError();
    if (launch_err != hipSuccess) return 343;

    if (use_gfx12_acc) {
        hipLaunchKernelGGL(
            supersonic_qwen35_reduce_m16_tile_argmax_tile_major_kernel,
            dim3(16, grid_z),
            dim3(256),
            0,
            0,
            batch_elems,
            tiles,
            static_cast<const float*>(block_best_vals),
            static_cast<const uint32_t*>(block_best_indices),
            static_cast<uint32_t*>(out_indices));
    } else {
        hipLaunchKernelGGL(
            supersonic_qwen35_reduce_m16_tile_argmax_kernel,
            dim3(16, grid_z),
            dim3(256),
            0,
            0,
            batch_elems,
            tiles,
            static_cast<const float*>(block_best_vals),
            static_cast<const uint32_t*>(block_best_indices),
            static_cast<uint32_t*>(out_indices));
    }
    launch_err = hipGetLastError();
    hipError_t sync_err = maybe_sync();
    if (launch_err != hipSuccess) return 344;
    if (sync_err != hipSuccess) return 345;
    return 0;
}

extern "C" int supersonic_qwen35_4b_hip_device_supports_wmma_i8(
    size_t device_ordinal,
    int* out_supported) {
    if (out_supported == nullptr) return 310;
    *out_supported = device_supports_wmma_i8(static_cast<int>(device_ordinal)) ? 1 : 0;
    return 0;
}

extern "C" int supersonic_qwen35_4b_hip_matmul_int4_dequant(
    int dtype,
    size_t device_ordinal,
    size_t batch_elems,
    int m, int n, int k,
    const void* lhs,
    const void* rhs_int4,
    const void* scale,
    const void* zero,
    const void* awq_inv_scale,
    int group_size,
    int quant_type,
    void* out) {
    constexpr int QWEN35_LOWBIT_NATIVE_INT4 = 4;
    constexpr int QWEN35_LOWBIT_GGML_Q8_0 = 8;
    constexpr int QWEN35_LOWBIT_GGML_Q4_K = 12;
    constexpr int QWEN35_LOWBIT_GGML_Q5_K = 13;
    constexpr int QWEN35_LOWBIT_GGML_Q6_K = 14;
    const bool native_int4 = quant_type == QWEN35_LOWBIT_NATIVE_INT4;
    const bool ggml_k =
        quant_type == QWEN35_LOWBIT_GGML_Q8_0 ||
        quant_type == QWEN35_LOWBIT_GGML_Q4_K ||
        quant_type == QWEN35_LOWBIT_GGML_Q5_K ||
        quant_type == QWEN35_LOWBIT_GGML_Q6_K;
    if (!native_int4 && !ggml_k) {
        return 273;
    }

    switch (dtype) {
    case 1: {
        // F32 output for the DFlash2 draft forward. The scalar kernel
        // dequantizes Q8_0 to F32, reads an F32 lhs, accumulates in F32, and
        // stores F32 — matching the upstream ggml F32 compute type so draft
        // activations never pass through a BF16 truncation. The WMMA path
        // stores BF16 only, so F32 output uses the scalar kernel. The caller
        // must supply an F32 lhs (the scalar kernel reads `lhs` as
        // `const float*`); a BF16 lhs would be reinterpreted as garbage.
        return matmul_int4_dequant_device<float>(
            static_cast<int>(device_ordinal), batch_elems, m, n, k,
            lhs, rhs_int4, scale, zero, awq_inv_scale, group_size, quant_type, out);
    }
    case 2: {
        // The tiled WMMA kernel fetches one (scale, zero) pair per BK-wide
        // K slab per row, so it's only correct when every BK-aligned slab
        // stays inside a single quantization group. That requires
        // group_size to be a multiple of TILED_WMMA_BK (= 64). The shipped
        // GPTQ bakes use group_size=128, so this path activates in
        // practice; other group sizes fall back to the scalar kernel.
        constexpr int TILED_BK = 64;
        if (device_supports_wmma_bf16(static_cast<int>(device_ordinal)) &&
            ((native_int4 && group_size % TILED_BK == 0) || ggml_k)) {
            return matmul_int4_dequant_wmma_bf16_device(
                static_cast<int>(device_ordinal), batch_elems, m, n, k,
                lhs, rhs_int4, scale, zero, awq_inv_scale, group_size, quant_type, out);
        }
        return matmul_int4_dequant_device<hip_bfloat16>(
            static_cast<int>(device_ordinal), batch_elems, m, n, k,
            lhs, rhs_int4, scale, zero, awq_inv_scale, group_size, quant_type, out);
    }
    default:
        return 272;
    }
}

extern "C" int supersonic_qwen35_4b_hip_matmul_int4_dequant_residual_add(
    int dtype,
    size_t device_ordinal,
    size_t batch_elems,
    int m, int n, int k,
    const void* lhs,
    const void* rhs_int4,
    const void* scale,
    const void* zero,
    const void* awq_inv_scale,
    int group_size,
    int quant_type,
    const void* residual,
    void* out) {
    (void)scale;
    (void)zero;
    (void)group_size;
    switch (dtype) {
    case 2:
        return matmul_int4_dequant_residual_add_bf16_device(
            static_cast<int>(device_ordinal), batch_elems, m, n, k,
            lhs, rhs_int4, awq_inv_scale, quant_type, residual, out);
    default:
        return 319;
    }
}

extern "C" int supersonic_qwen35_4b_hip_matmul_ggml_pair_dequant(
    int dtype,
    size_t device_ordinal,
    size_t batch_elems,
    int m, int n_each, int k,
    const void* lhs,
    const void* rhs_first,
    const void* rhs_second,
    int quant_type,
    void* out) {
    switch (dtype) {
    case 2:
        return matmul_ggml_pair_wmma_bf16_device(
            static_cast<int>(device_ordinal), batch_elems, m, n_each, k,
            lhs, rhs_first, rhs_second, quant_type, out);
    default:
        return 298;
    }
}

extern "C" int supersonic_qwen35_4b_hip_matmul_ggml_pair_swiglu(
    int dtype,
    size_t device_ordinal,
    size_t batch_elems,
    int m, int n_each, int k,
    const void* lhs,
    const void* rhs_gate,
    const void* rhs_up,
    int quant_type,
    void* out) {
    switch (dtype) {
    case 2:
        return matmul_ggml_pair_swiglu_wmma_bf16_device(
            static_cast<int>(device_ordinal), batch_elems, m, n_each, k,
            lhs, rhs_gate, rhs_up, quant_type, out);
    default:
        return 327;
    }
}

extern "C" int supersonic_qwen35_4b_hip_quantize_mmq_q8_1(
    int dtype,
    size_t device_ordinal,
    size_t batch_elems,
    int m,
    int k,
    const void* lhs,
    int quant_type,
    void* out) {
    switch (dtype) {
    case 1:
        return quantize_mmq_q8_1_device<float>(
            static_cast<int>(device_ordinal), batch_elems, m, k, lhs, quant_type, out);
    case 2:
        return quantize_mmq_q8_1_device<hip_bfloat16>(
            static_cast<int>(device_ordinal), batch_elems, m, k, lhs, quant_type, out);
    default:
        return 304;
    }
}

extern "C" int supersonic_qwen35_4b_hip_matmul_mmq_q8_1_q6_k(
    int dtype,
    size_t device_ordinal,
    size_t batch_elems,
    int m,
    int n,
    int k,
    const void* q8,
    const void* rhs_q6,
    void* out) {
    switch (dtype) {
    case 2:
        return matmul_mmq_q8_1_q6_k_device(
            static_cast<int>(device_ordinal), batch_elems, m, n, k, q8, rhs_q6, nullptr, out);
    default:
        return 308;
    }
}

extern "C" int supersonic_qwen35_4b_hip_matmul_mmq_q8_1_q6_k_residual_add(
    int dtype,
    size_t device_ordinal,
    size_t batch_elems,
    int m,
    int n,
    int k,
    const void* q8,
    const void* rhs_q6,
    const void* residual,
    void* out) {
    switch (dtype) {
    case 2:
        return matmul_mmq_q8_1_q6_k_device(
            static_cast<int>(device_ordinal), batch_elems, m, n, k, q8, rhs_q6, residual, out);
    default:
        return 330;
    }
}

extern "C" int supersonic_qwen35_4b_hip_matmul_q6_k_m16_argmax(
    int dtype,
    size_t device_ordinal,
    size_t batch_elems,
    int m,
    int n,
    int k,
    const void* lhs,
    const void* rhs_q6,
    void* block_best_vals,
    void* block_best_indices,
    void* out_indices) {
    switch (dtype) {
    case 2:
        return matmul_q6_k_m16_argmax_device(
            static_cast<int>(device_ordinal), batch_elems, m, n, k,
            lhs, rhs_q6, block_best_vals, block_best_indices, out_indices);
    default:
        return 346;
    }
}

extern "C" int supersonic_qwen38_hip_q6_k_scalar_head_f32(
    int dtype,
    size_t device_ordinal,
    size_t lhs_elems,
    size_t rhs_bytes,
    size_t out_elems,
    const void* lhs,
    const void* rhs_q6,
    void* out,
    size_t row_start,
    size_t row_count) {
    constexpr size_t LHS_ELEMS = 5120;
    constexpr size_t ROWS = 248320;
    constexpr size_t ROW_BYTES = 20 * 210;
    constexpr size_t RHS_BYTES = ROWS * ROW_BYTES;

    if (dtype != 2) return 360;
    if (lhs_elems != LHS_ELEMS || rhs_bytes != RHS_BYTES || out_elems != ROWS) return 361;
    if (lhs == nullptr || rhs_q6 == nullptr || out == nullptr) return 362;
    if (row_count == 0) return 363;
    if (row_start > ROWS || row_count > ROWS - row_start) return 364;
    if (device_ordinal > 0x7fffffffu) return 365;

    const int ordinal = static_cast<int>(device_ordinal);
    ScopedHipDevice scoped(ordinal);
    if (!scoped.ok()) return prefill_backend_failure(365, scoped.status);

    hipDeviceProp_t props;
    const hipError_t props_err = hipGetDeviceProperties(&props, ordinal);
    if (props_err != hipSuccess) return prefill_backend_failure(365, props_err);
    if (std::strcmp(props.gcnArchName, "gfx1201") != 0) return 366;

    constexpr int THREADS = 128;
    constexpr int ROWS_PER_CTA = 4;
    const unsigned int blocks =
        static_cast<unsigned int>((row_count + ROWS_PER_CTA - 1) / ROWS_PER_CTA);
    hipLaunchKernelGGL(
        supersonic_qwen38_q6_k_scalar_head_f32_kernel,
        dim3(blocks),
        dim3(THREADS),
        0,
        0,
        static_cast<const uint16_t*>(lhs),
        static_cast<const uint8_t*>(rhs_q6),
        static_cast<float*>(out),
        static_cast<int>(row_start),
        static_cast<int>(row_count));
    const hipError_t launch_err = hipGetLastError();
    const hipError_t sync_err = maybe_sync();
    if (launch_err != hipSuccess) return prefill_backend_failure(368, launch_err);
    if (sync_err != hipSuccess) return prefill_backend_failure(369, sync_err);
    return 0;
}

template <typename T>
int int4_sparse_outlier_add_device(
    int device_ordinal,
    int rows,
    int n,
    int k,
    int sub_cols,
    const void* lhs,
    const void* outlier_cols,
    const void* outlier_delta,
    void* out
) {
    ScopedHipDevice scoped(device_ordinal);
    constexpr int block = 256;
    const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(n);
    const unsigned int grid = static_cast<unsigned int>((total + block - 1) / block);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_int4_sparse_outlier_add_kernel<T>),
        dim3(grid),
        dim3(block),
        0,
        0,
        rows,
        n,
        k,
        sub_cols,
        static_cast<const T*>(lhs),
        static_cast<const uint32_t*>(outlier_cols),
        static_cast<const T*>(outlier_delta),
        static_cast<T*>(out));
    hipError_t launch_err = hipGetLastError();
    hipError_t sync_err = maybe_sync();
    if (launch_err != hipSuccess) return 294;
    if (sync_err != hipSuccess) return 295;
    return 0;
}

extern "C" int supersonic_qwen35_4b_hip_int4_sparse_outlier_add(
    int dtype,
    size_t device_ordinal,
    int rows,
    int n,
    int k,
    int sub_cols,
    const void* lhs,
    const void* outlier_cols,
    const void* outlier_delta,
    void* out
) {
    switch (dtype) {
    case 2:
        return int4_sparse_outlier_add_device<hip_bfloat16>(
            static_cast<int>(device_ordinal), rows, n, k, sub_cols,
            lhs, outlier_cols, outlier_delta, out);
    default:
        return 296;
    }
}

extern "C" int supersonic_qwen35_4b_hip_cast(
    int input_dtype,
    int output_dtype,
    size_t device_ordinal,
    size_t total_elems,
    const void* xs,
    void* out) {
    switch (input_dtype) {
    case 0:
        switch (output_dtype) {
        case 0:
            return cast_device<half, half>(static_cast<int>(device_ordinal), static_cast<int>(total_elems), xs, out);
        case 1:
            return cast_device<half, float>(static_cast<int>(device_ordinal), static_cast<int>(total_elems), xs, out);
        case 2:
            return cast_device<half, hip_bfloat16>(static_cast<int>(device_ordinal), static_cast<int>(total_elems), xs, out);
        default:
            return 137;
        }
    case 1:
        switch (output_dtype) {
        case 0:
            return cast_device<float, half>(static_cast<int>(device_ordinal), static_cast<int>(total_elems), xs, out);
        case 1:
            return cast_device<float, float>(static_cast<int>(device_ordinal), static_cast<int>(total_elems), xs, out);
        case 2:
            return cast_device<float, hip_bfloat16>(static_cast<int>(device_ordinal), static_cast<int>(total_elems), xs, out);
        default:
            return 137;
        }
    case 2:
        switch (output_dtype) {
        case 0:
            return cast_device<hip_bfloat16, half>(static_cast<int>(device_ordinal), static_cast<int>(total_elems), xs, out);
        case 1:
            return cast_device<hip_bfloat16, float>(static_cast<int>(device_ordinal), static_cast<int>(total_elems), xs, out);
        case 2:
            return cast_device<hip_bfloat16, hip_bfloat16>(static_cast<int>(device_ordinal), static_cast<int>(total_elems), xs, out);
        default:
            return 137;
        }
    default:
        return 135;
    }
}

extern "C" int supersonic_qwen35_4b_hip_binary_broadcast(
    int op,
    int dtype,
    size_t device_ordinal,
    int rank,
    size_t total_elems,
    const void* lhs,
    const void* rhs,
    const int* lhs_strides,
    const int* rhs_strides,
    const int* out_dims,
    void* out) {
    switch (dtype) {
    case 0:
        return binary_broadcast_device<half>(
            op,
            static_cast<int>(device_ordinal),
            rank,
            total_elems,
            lhs,
            rhs,
            lhs_strides,
            rhs_strides,
            out_dims,
            out);
    case 1:
        return binary_broadcast_device<float>(
            op,
            static_cast<int>(device_ordinal),
            rank,
            total_elems,
            lhs,
            rhs,
            lhs_strides,
            rhs_strides,
            out_dims,
            out);
    case 2:
        return binary_broadcast_device<hip_bfloat16>(
            op,
            static_cast<int>(device_ordinal),
            rank,
            total_elems,
            lhs,
            rhs,
            lhs_strides,
            rhs_strides,
            out_dims,
            out);
    default:
        return 140;
    }
}

extern "C" int supersonic_qwen35_4b_hip_batched_matmul(
    int dtype,
    size_t device_ordinal,
    int batch_rank,
    size_t batch_elems,
    int m,
    int n,
    int k,
    const int* lhs_batch_dims,
    const int* rhs_batch_dims,
    const int* out_batch_dims,
    const void* lhs,
    const void* rhs,
    void* out
) {
    switch (dtype) {
    case 0:
        return batched_matmul_device<half>(
            static_cast<int>(device_ordinal),
            batch_rank,
            batch_elems,
            m,
            n,
            k,
            lhs_batch_dims,
            rhs_batch_dims,
            out_batch_dims,
            lhs,
            rhs,
            out);
    case 1:
        return batched_matmul_device<float>(
            static_cast<int>(device_ordinal),
            batch_rank,
            batch_elems,
            m,
            n,
            k,
            lhs_batch_dims,
            rhs_batch_dims,
            out_batch_dims,
            lhs,
            rhs,
            out);
    case 2:
        return batched_matmul_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            batch_rank,
            batch_elems,
            m,
            n,
            k,
            lhs_batch_dims,
            rhs_batch_dims,
            out_batch_dims,
            lhs,
            rhs,
            out);
    default:
        return 144;
    }
}

extern "C" int supersonic_qwen35_4b_hip_mul_scalar(
    int dtype,
    size_t device_ordinal,
    size_t total_elems,
    float scalar,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return mul_scalar_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            scalar,
            xs,
            out);
    case 1:
        return mul_scalar_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            scalar,
            xs,
            out);
    case 2:
        return mul_scalar_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            scalar,
            xs,
            out);
    default:
        return 147;
    }
}

extern "C" int supersonic_qwen35_4b_hip_reduce_keepdim(
    int dtype,
    size_t device_ordinal,
    size_t outer,
    size_t reduce,
    size_t inner,
    int sum,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return reduce_keepdim_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(outer),
            static_cast<int>(reduce),
            static_cast<int>(inner),
            sum != 0,
            xs,
            out);
    case 1:
        return reduce_keepdim_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(outer),
            static_cast<int>(reduce),
            static_cast<int>(inner),
            sum != 0,
            xs,
            out);
    case 2:
        return reduce_keepdim_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(outer),
            static_cast<int>(reduce),
            static_cast<int>(inner),
            sum != 0,
            xs,
            out);
    default:
        return 149;
    }
}

extern "C" int supersonic_qwen35_4b_hip_add_scalar(
    int dtype,
    size_t device_ordinal,
    size_t total_elems,
    float scalar,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return add_scalar_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            scalar,
            xs,
            out);
    case 1:
        return add_scalar_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            scalar,
            xs,
            out);
    case 2:
        return add_scalar_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            scalar,
            xs,
            out);
    default:
        return 153;
    }
}

extern "C" int supersonic_qwen35_4b_hip_sqrt(
    int dtype,
    size_t device_ordinal,
    size_t total_elems,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return sqrt_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    case 1:
        return sqrt_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    case 2:
        return sqrt_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            xs,
            out);
    default:
        return 154;
    }
}

extern "C" int supersonic_qwen35_4b_hip_l2norm(
    int dtype,
    size_t device_ordinal,
    size_t n_rows,
    size_t n_cols,
    float eps,
    const void* xs,
    void* out) {
    switch (dtype) {
    case 0:
        return l2norm_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(n_rows),
            static_cast<int>(n_cols),
            eps,
            xs,
            out);
    case 1:
        return l2norm_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(n_rows),
            static_cast<int>(n_cols),
            eps,
            xs,
            out);
    case 2:
        return l2norm_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(n_rows),
            static_cast<int>(n_cols),
            eps,
            xs,
            out);
    default:
        return 92;
    }
}

extern "C" int supersonic_qwen35_4b_hip_value_decay(
    int dtype,
    size_t device_ordinal,
    size_t total_elems,
    size_t num_heads,
    const void* a,
    const void* dt_bias,
    const void* a_log_exp,
    void* out) {
    switch (dtype) {
    case 0:
        return value_decay_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            static_cast<int>(num_heads),
            a,
            dt_bias,
            a_log_exp,
            out);
    case 1:
        return value_decay_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            static_cast<int>(num_heads),
            a,
            dt_bias,
            a_log_exp,
            out);
    case 2:
        return value_decay_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(total_elems),
            static_cast<int>(num_heads),
            a,
            dt_bias,
            a_log_exp,
            out);
    default:
        return 95;
    }
}

extern "C" int supersonic_qwen35_4b_hip_rms_norm(
    int dtype,
    size_t device_ordinal,
    size_t n_rows,
    size_t n_cols,
    float eps,
    int add_unit_offset,
    const void* xs,
    const void* weight,
    void* out) {
    switch (dtype) {
    case 0:
        return add_unit_offset
            ? rms_norm_device<half, true>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(n_rows),
                  static_cast<int>(n_cols),
                  eps,
                  xs,
                  weight,
                  out)
            : rms_norm_device<half, false>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(n_rows),
                  static_cast<int>(n_cols),
                  eps,
                  xs,
                  weight,
                  out);
    case 1:
        return add_unit_offset
            ? rms_norm_device<float, true>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(n_rows),
                  static_cast<int>(n_cols),
                  eps,
                  xs,
                  weight,
                  out)
            : rms_norm_device<float, false>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(n_rows),
                  static_cast<int>(n_cols),
                  eps,
                  xs,
                  weight,
                  out);
    case 2:
        return add_unit_offset
            ? rms_norm_device<hip_bfloat16, true>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(n_rows),
                  static_cast<int>(n_cols),
                  eps,
                  xs,
                  weight,
                  out)
            : rms_norm_device<hip_bfloat16, false>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(n_rows),
                  static_cast<int>(n_cols),
                  eps,
                  xs,
                  weight,
                  out);
    default:
        return 74;
    }
}

extern "C" int supersonic_qwen35_4b_hip_fused_rms_norm_linear(
    int dtype,
    size_t device_ordinal,
    size_t hidden_dim,
    size_t out_dim,
    float eps,
    int add_unit_offset,
    const void* hidden,
    const void* norm_weight,
    const void* proj_weight,
    void* out) {
    switch (dtype) {
    case 0:
        return add_unit_offset
            ? fused_rms_norm_linear_device<half, true>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(hidden_dim),
                  static_cast<int>(out_dim),
                  eps,
                  hidden,
                  norm_weight,
                  proj_weight,
                  out)
            : fused_rms_norm_linear_device<half, false>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(hidden_dim),
                  static_cast<int>(out_dim),
                  eps,
                  hidden,
                  norm_weight,
                  proj_weight,
                  out);
    case 1:
        return add_unit_offset
            ? fused_rms_norm_linear_device<float, true>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(hidden_dim),
                  static_cast<int>(out_dim),
                  eps,
                  hidden,
                  norm_weight,
                  proj_weight,
                  out)
            : fused_rms_norm_linear_device<float, false>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(hidden_dim),
                  static_cast<int>(out_dim),
                  eps,
                  hidden,
                  norm_weight,
                  proj_weight,
                  out);
    case 2:
        return add_unit_offset
            ? fused_rms_norm_linear_device<hip_bfloat16, true>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(hidden_dim),
                  static_cast<int>(out_dim),
                  eps,
                  hidden,
                  norm_weight,
                  proj_weight,
                  out)
            : fused_rms_norm_linear_device<hip_bfloat16, false>(
                  static_cast<int>(device_ordinal),
                  static_cast<int>(hidden_dim),
                  static_cast<int>(out_dim),
                  eps,
                  hidden,
                  norm_weight,
                  proj_weight,
                  out);
    default:
        return 132;
    }
}

extern "C" int supersonic_qwen35_4b_hip_rms_norm_gated(
    int dtype,
    size_t device_ordinal,
    size_t n_rows,
    size_t n_cols,
    float eps,
    const void* hidden,
    const void* gate,
    const void* weight,
    void* out) {
    switch (dtype) {
    case 0:
        return rms_norm_gated_device<half>(
            static_cast<int>(device_ordinal),
            static_cast<int>(n_rows),
            static_cast<int>(n_cols),
            eps,
            hidden,
            gate,
            weight,
            out);
    case 1:
        return rms_norm_gated_device<float>(
            static_cast<int>(device_ordinal),
            static_cast<int>(n_rows),
            static_cast<int>(n_cols),
            eps,
            hidden,
            gate,
            weight,
            out);
    case 2:
        return rms_norm_gated_device<hip_bfloat16>(
            static_cast<int>(device_ordinal),
            static_cast<int>(n_rows),
            static_cast<int>(n_cols),
            eps,
            hidden,
            gate,
            weight,
            out);
    default:
        return 84;
    }
}

template <typename T>
int mlp_decode_megakernel_device(
    int device_ordinal,
    int hidden_dim,
    int intermediate_size,
    float norm_eps,
    const void* hidden_in,
    const void* norm_weight,
    const void* gate_proj_w,
    const void* up_proj_w,
    const void* down_proj_w,
    float* gate_up_scratch,
    void* hidden_out,
    unsigned int* row_counter
) {
    ScopedHipDevice scoped(device_ordinal);

    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_ordinal) != hipSuccess) return 200;

    const int num_blocks = props.multiProcessorCount > 0 ? props.multiProcessorCount : 16;
    constexpr int block_size = 256;
    const size_t shared_bytes =
        static_cast<size_t>(hidden_dim) * sizeof(float) * 2 +  // hidden + normed
        block_size * sizeof(float);                              // scratch

    // --- Phase 1: RMSNorm + gate/up projections ---
    unsigned int zero = 0;
    if (hipMemcpy(row_counter, &zero, sizeof(unsigned int), hipMemcpyHostToDevice) != hipSuccess)
        return 201;
    if (maybe_sync() != hipSuccess) return 202;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_mlp_decode_megakernel<T>),
        dim3(static_cast<unsigned int>(num_blocks)),
        dim3(block_size),
        shared_bytes,
        0,
        hidden_dim,
        intermediate_size,
        norm_eps,
        static_cast<const T*>(hidden_in),
        static_cast<const T*>(norm_weight),
        static_cast<const T*>(gate_proj_w),
        static_cast<const T*>(up_proj_w),
        static_cast<const T*>(down_proj_w),
        gate_up_scratch,
        static_cast<T*>(hidden_out),
        row_counter);
    if (hipGetLastError() != hipSuccess) return 203;
    if (maybe_sync() != hipSuccess) return 204;

    // --- Phase 2: SwiGLU activation ---
    {
        constexpr int swiglu_block = 256;
        const unsigned int swiglu_grid =
            static_cast<unsigned int>((intermediate_size + swiglu_block - 1) / swiglu_block);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(supersonic_qwen35_mlp_swiglu_kernel<T>),
            dim3(swiglu_grid),
            dim3(swiglu_block),
            0, 0,
            intermediate_size,
            gate_up_scratch);
        if (hipGetLastError() != hipSuccess) return 205;
        if (maybe_sync() != hipSuccess) return 206;
    }

    // --- Phase 3: down_proj matvec ---
    if (hipMemcpy(row_counter, &zero, sizeof(unsigned int), hipMemcpyHostToDevice) != hipSuccess)
        return 207;
    if (maybe_sync() != hipSuccess) return 208;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_mlp_down_proj_kernel<T>),
        dim3(static_cast<unsigned int>(num_blocks)),
        dim3(block_size),
        block_size * sizeof(float),
        0,
        hidden_dim,
        intermediate_size,
        static_cast<const T*>(down_proj_w),
        gate_up_scratch,
        static_cast<T*>(hidden_out),
        row_counter);
    if (hipGetLastError() != hipSuccess) return 209;
    if (maybe_sync() != hipSuccess) return 210;
    return 0;
}

extern "C" int supersonic_qwen35_4b_hip_mlp_decode_megakernel(
    int dtype,
    size_t device_ordinal,
    size_t hidden_dim,
    size_t intermediate_size,
    float norm_eps,
    const void* hidden_in,
    const void* norm_weight,
    const void* gate_proj_w,
    const void* up_proj_w,
    const void* down_proj_w,
    float* gate_up_scratch,
    void* hidden_out,
    unsigned int* row_counter) {
    switch (dtype) {
    case 0:
        return mlp_decode_megakernel_device<half>(
            static_cast<int>(device_ordinal), static_cast<int>(hidden_dim),
            static_cast<int>(intermediate_size), norm_eps, hidden_in, norm_weight,
            gate_proj_w, up_proj_w, down_proj_w, gate_up_scratch, hidden_out, row_counter);
    case 2:
        return mlp_decode_megakernel_device<hip_bfloat16>(
            static_cast<int>(device_ordinal), static_cast<int>(hidden_dim),
            static_cast<int>(intermediate_size), norm_eps, hidden_in, norm_weight,
            gate_proj_w, up_proj_w, down_proj_w, gate_up_scratch, hidden_out, row_counter);
    default:
        return 205;
    }
}

template <typename T>
int norm_multi_proj_device(
    int device_ordinal,
    int hidden_dim,
    int total_rows,
    float norm_eps,
    const void* hidden_in,
    const void* norm_weight,
    const Qwen35ProjectionDesc* proj_table,
    int num_projections,
    float* output,
    unsigned int* row_counter
) {
    ScopedHipDevice scoped(device_ordinal);

    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_ordinal) != hipSuccess) return 220;

    const int num_blocks = props.multiProcessorCount > 0 ? props.multiProcessorCount : 16;
    constexpr int block_size = 256;
    const size_t shared_bytes =
        static_cast<size_t>(hidden_dim) * sizeof(float) * 2 + block_size * sizeof(float);

    unsigned int zero = 0;
    if (hipMemcpy(row_counter, &zero, sizeof(unsigned int), hipMemcpyHostToDevice) != hipSuccess)
        return 221;
    if (maybe_sync() != hipSuccess) return 222;

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_norm_multi_proj_kernel<T>),
        dim3(static_cast<unsigned int>(num_blocks)),
        dim3(block_size),
        shared_bytes,
        0,
        hidden_dim,
        total_rows,
        norm_eps,
        static_cast<const T*>(hidden_in),
        static_cast<const T*>(norm_weight),
        proj_table,
        num_projections,
        output,
        row_counter);
    if (hipGetLastError() != hipSuccess) return 223;
    if (maybe_sync() != hipSuccess) return 224;
    return 0;
}

extern "C" int supersonic_qwen35_4b_hip_norm_multi_proj(
    int dtype,
    size_t device_ordinal,
    size_t hidden_dim,
    size_t total_rows,
    float norm_eps,
    const void* hidden_in,
    const void* norm_weight,
    const void* proj_table,       // Qwen35ProjectionDesc* on device
    size_t num_projections,
    float* output,
    unsigned int* row_counter) {
    switch (dtype) {
    case 0:
        return norm_multi_proj_device<half>(
            static_cast<int>(device_ordinal), static_cast<int>(hidden_dim),
            static_cast<int>(total_rows), norm_eps, hidden_in, norm_weight,
            static_cast<const Qwen35ProjectionDesc*>(proj_table),
            static_cast<int>(num_projections), output, row_counter);
    case 2:
        return norm_multi_proj_device<hip_bfloat16>(
            static_cast<int>(device_ordinal), static_cast<int>(hidden_dim),
            static_cast<int>(total_rows), norm_eps, hidden_in, norm_weight,
            static_cast<const Qwen35ProjectionDesc*>(proj_table),
            static_cast<int>(num_projections), output, row_counter);
    default:
        return 225;
    }
}

// Standalone work-stealing matvec: out[out_dim] = W[out_dim, in_dim] × input[in_dim]
// Reuses the down_proj kernel pattern for arbitrary matvec.
template <typename T>
int standalone_matvec_device(
    int device_ordinal,
    int in_dim,
    int out_dim,
    const void* input,       // [in_dim] F32
    const void* weight,      // [out_dim, in_dim] BF16
    void* output,            // [out_dim] BF16
    unsigned int* row_counter
) {
    ScopedHipDevice scoped(device_ordinal);

    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_ordinal) != hipSuccess) return 230;

    const int num_blocks = props.multiProcessorCount > 0 ? props.multiProcessorCount : 16;
    constexpr int block_size = 256;

    unsigned int zero = 0;
    if (hipMemcpy(row_counter, &zero, sizeof(unsigned int), hipMemcpyHostToDevice) != hipSuccess)
        return 231;
    if (maybe_sync() != hipSuccess) return 232;

    const size_t shared_bytes = block_size * sizeof(float);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(supersonic_qwen35_standalone_matvec_kernel<T>),
        dim3(static_cast<unsigned int>(num_blocks)),
        dim3(block_size),
        shared_bytes,
        0,
        out_dim,
        in_dim,
        static_cast<const T*>(weight),
        static_cast<const T*>(input),
        static_cast<T*>(output),
        row_counter);
    if (hipGetLastError() != hipSuccess) return 233;
    if (maybe_sync() != hipSuccess) return 234;
    return 0;
}

extern "C" int supersonic_qwen35_4b_hip_standalone_matvec(
    int dtype,
    size_t device_ordinal,
    size_t in_dim,
    size_t out_dim,
    const void* input,
    const void* weight,
    void* output,
    unsigned int* row_counter) {
    switch (dtype) {
    case 0:
        return standalone_matvec_device<half>(
            static_cast<int>(device_ordinal), static_cast<int>(in_dim),
            static_cast<int>(out_dim), input, weight, output, row_counter);
    case 2:
        return standalone_matvec_device<hip_bfloat16>(
            static_cast<int>(device_ordinal), static_cast<int>(in_dim),
            static_cast<int>(out_dim), input, weight, output, row_counter);
    default:
        return 235;
    }
}

// Once persistent decode has enqueued work that references model-owned
// buffers, a launch or owning-device synchronization failure is unrecoverable:
// returning an ordinary status would let Rust drop those buffers while the
// device may still dereference them. Keep this check separate from validation
// and allocation failures, which remain ordinary return paths.
int persistent_decode_post_enqueue_status(
    hipError_t launch_err, hipError_t sync_err, int device_ordinal) {
    if (launch_err != hipSuccess) {
        std::fprintf(
            stderr,
            "[decode] persistent launch failure status=%d ordinal=%d\n",
            static_cast<int>(launch_err),
            device_ordinal);
        supersonic_gpu_integrity_fail_stop(
            "persistent decode launch", static_cast<int>(launch_err), device_ordinal);
    }
    if (sync_err != hipSuccess) {
        std::fprintf(
            stderr,
            "[decode] persistent synchronize failure status=%d ordinal=%d\n",
            static_cast<int>(sync_err),
            device_ordinal);
        supersonic_gpu_integrity_fail_stop(
            "persistent decode synchronize", static_cast<int>(sync_err), device_ordinal);
    }
    return 0;
}

int persistent_decode_prepare_only_status(hipError_t sync_err, int device_ordinal) {
    if (sync_err != hipSuccess) {
        std::fprintf(
            stderr,
            "[decode] persistent prepare-only synchronize failure status=%d ordinal=%d\n",
            static_cast<int>(sync_err),
            device_ordinal);
        supersonic_gpu_integrity_fail_stop(
            "persistent decode prepare-only synchronize",
            static_cast<int>(sync_err),
            device_ordinal);
    }
    return 0;
}

#ifdef SUPERSONIC_FAILURE_INJECTION
extern "C" void supersonic_qwen35_4b_test_trigger_persistent_decode_failure(
    int launch_status, int sync_status) {
    (void)persistent_decode_post_enqueue_status(
        static_cast<hipError_t>(launch_status),
        static_cast<hipError_t>(sync_status),
        0);
}

extern "C" void supersonic_qwen35_4b_test_trigger_prepare_only_failure(
    int sync_status) {
    (void)persistent_decode_prepare_only_status(
        static_cast<hipError_t>(sync_status), 0);
}
#endif

template <typename T>
int persistent_decode_device(
    int device_ordinal,
    int num_layers,
    int hidden_dim,
    int intermediate_size,
    int seqlen_offset,
    const void* layers,
    void* hidden_io,
    float* workspace,
    unsigned int* counters,
    unsigned int* barrier_counter,
    unsigned int* barrier_flag,
    unsigned long long* timing_slots,
    const void* cos_table,
    const void* sin_table,
    int rotary_dim,
    int proj_buf_floats,
    int attn_scratch_floats,
    int enable_attention_trace,
    const void* fp8_scales,
    const void* kv_fp8_descs,
    int batch_size,
    const void* batch_descs,
    const void* int4_scales
) {
    // Dynamic KV quantization is not part of the Qwen3.8 product path. Reject
    // a non-null descriptor at the ABI boundary before any device work or
    // process-global decode state can be touched.
    if (kv_fp8_descs != nullptr) {
        return 256;
    }
    DecodeBridgeLockGuard guard;
    ScopedHipDevice scoped(device_ordinal);
    if (!scoped.ok()) {
        return prefill_backend_failure(249, scoped.status);
    }
    const int gemm_flush_status = supersonic_gqh_hip_gemm_flush(device_ordinal);
    if (gemm_flush_status != 0) {
        return prefill_backend_failure(
            250, static_cast<hipError_t>(gemm_flush_status));
    }

    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device_ordinal) != hipSuccess) return 250;

    // rocprofv3 on gfx1150 revealed `multiProcessorCount` reports 8 (WGP
    // count on RDNA3, not CU count) while the device has 16 CUs — the
    // original 1x grid left half the GPU idle with only 1 block/CU, and
    // register pressure (VGPR=128) prevents a second resident block.
    // Oversubscribing 2x on RDNA3 (= one block per CU) is a proven safe
    // win: ~1.57x faster on qwen3.8-27b BF16, no hangs across tested
    // Qwen variants.
    //
    // HIP docs note `multiProcessorCount` reports CUs on CDNA (MI-series)
    // and WGPs on RDNA. On arches where it already reports CU count, 2x
    // would over-subscribe and can deadlock via `grid_barrier` when only
    // one block/CU is resident. Restrict the default multiplier to the
    // arches we've actually validated (gfx11xx RDNA3/3.5 at time of
    // writing); other arches get the conservative 1x default.
    //
    // Higher multipliers (3x+) hang silently on models with more
    // transformer layers (the retained Qwen3.8 geometry at 4x produces no output) —
    // suspected grid_barrier scaling issue. Env var
    // Grid-size priority (first match wins):
    //   1. SUPERSONIC_QWEN4B_BLOCKS env var (explicit user override).
    //   2. Per-model preset set via `supersonic_qwen35_4b_hip_set_launch_preset`
    //      from the Rust registry (e.g. 0.8B gets 32 + cooperative).
    //   3. 2x multiProcessorCount default on RDNA3/gfx11xx (empirically
    //      safe at non-cooperative launch on every tested Qwen variant).
    //
    // Cooperative launch is enabled when SUPERSONIC_QWEN4B_COOP is set OR
    // when the active preset opts in. The homebrew `grid_barrier` assumes
    // every block is co-resident; cooperative launch enforces that and
    // fails cleanly on over-subscription instead of deadlocking.
    //
    // Why cooperative is opt-in rather than always-on: `hipOccupancyMax-
    // ActiveBlocksPerMultiprocessor` is strictly conservative — on 4B it
    // reports 1 block/MP while non-cooperative launch empirically handles
    // 2. Cooperative-by-default would regress 4B throughput.
    int num_blocks = props.multiProcessorCount > 0 ? props.multiProcessorCount : 16;
    int preset_blocks = 0, preset_coop = 0;
    qwen4b_get_launch_preset(preset_blocks, preset_coop);
    bool preset_coop_hint = false;
    bool user_grid = false;
    if (const char* bs_env = std::getenv("SUPERSONIC_QWEN4B_BLOCKS")) {
        int override_val = std::atoi(bs_env);
        if (override_val > 0) {
            num_blocks = override_val;
            user_grid = true;
        }
    } else if (preset_blocks > 0) {
        num_blocks = preset_blocks;
        preset_coop_hint = preset_coop != 0;
    }
    const bool coop_requested =
        std::getenv("SUPERSONIC_QWEN4B_COOP") != nullptr || preset_coop_hint;
    constexpr int block_size = 256;
    const size_t fp8_lut_size =
        (fp8_scales != nullptr || kv_fp8_descs != nullptr) ? 256u : 0u;
    const size_t shared_bytes = (block_size + fp8_lut_size) * sizeof(float);

    int coop_supported = 0;
    int max_blocks_per_mp = 0;
    const void* kernel_fp = reinterpret_cast<const void*>(
        &supersonic_qwen35_persistent_decode_kernel<T>);
    int api_occ = 0;
    (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &api_occ, kernel_fp, block_size, shared_bytes);
    hipFuncAttributes attr{};
    (void)hipFuncGetAttributes(&attr, kernel_fp);
    // gfx1100 WGP: 4 SIMDs × 1536 VGPR, wave32. Occupancy API is optimistic
    // once scratch/spills show up; size the default grid from VGPR+LDS so
    // the ticket barrier never launches more blocks than can be resident.
    int vgpr = attr.numRegs > 0 ? attr.numRegs : 256;
    int waves_simd = 1536 / vgpr;
    if (attr.localSizeBytes > 16 && waves_simd > 1) {
        waves_simd -= 1;
    }
    if (waves_simd < 1) waves_simd = 1;
    if (waves_simd > 16) waves_simd = 16;
    int from_vgpr = (waves_simd * 4) / (block_size / 32);
    if (from_vgpr < 1) from_vgpr = 1;
    int from_lds = static_cast<int>(65536 / (shared_bytes > 0 ? shared_bytes : 1));
    if (from_lds < 1) from_lds = 1;
    int safe_per_mp = from_vgpr < from_lds ? from_vgpr : from_lds;
    if (!user_grid && preset_blocks <= 0) {
        const char* arch = props.gcnArchName;
        const bool is_rdna3_wgp_arch =
            arch[0] == 'g' && arch[1] == 'f' && arch[2] == 'x' &&
            arch[3] == '1' && arch[4] == '1';
        if (is_rdna3_wgp_arch) {
            // Ticket barrier is fine at 144 on a skinny kernel, but this
            // 253-VGPR megakernel still deadlocks at 3x (only 2 blocks
            // actually stay resident). Keep the proven 2x default.
            int mult = safe_per_mp;
            if (mult > 2) mult = 2;
            if (mult < 1) mult = 1;
            num_blocks = props.multiProcessorCount * mult;
        }
    }
    {
        static bool dumped_occ = false;
        if (!dumped_occ) {
            dumped_occ = true;
            std::fprintf(
                stderr,
                "[hip-occ] persistent_decode hidden=%d B=%d grid=%d CUs=%d "
                "api_blocks/CU=%d safe_blocks/CU=%d vgpr=%d scratch=%zu "
                "static_lds=%zu dyn_lds=%zu max_threads=%d\n",
                hidden_dim,
                batch_size,
                num_blocks,
                props.multiProcessorCount,
                api_occ,
                safe_per_mp,
                attr.numRegs,
                attr.localSizeBytes,
                attr.sharedSizeBytes,
                shared_bytes,
                attr.maxThreadsPerBlock);
        }
    }
    if (coop_requested) {
        (void)hipDeviceGetAttribute(
            &coop_supported, hipDeviceAttributeCooperativeLaunch, device_ordinal);
        if (coop_supported) {
            if (hipOccupancyMaxActiveBlocksPerMultiprocessor(
                    &max_blocks_per_mp, kernel_fp, block_size, shared_bytes) !=
                hipSuccess) {
                max_blocks_per_mp = 0;
            }
            if (max_blocks_per_mp > 0) {
                int coop_max_grid = props.multiProcessorCount * max_blocks_per_mp;
                if (num_blocks > coop_max_grid) num_blocks = coop_max_grid;
            }
        }
    }

    // If the caller asked for cooperative launch but the device or runtime
    // can't actually provide it, refuse rather than fall back to the
    // non-cooperative path. A `SUPERSONIC_QWEN4B_BLOCKS=128` with `COOP=1`
    // expects the cooperative cap to keep it safe; silently running the
    // non-coop launcher with 128 blocks is exactly the grid_barrier
    // oversubscription hang the opt-in was designed to prevent.
    if (coop_requested && (!coop_supported || max_blocks_per_mp <= 0)) {
        return 261;
    }

    int io_flags = 3;
    const bool use_gqh_split = int4_scales != nullptr
        && hidden_dim >= 5120
        && std::getenv("SUPERSONIC_QWEN38_GQH_NOSPLIT") == nullptr
        && !coop_requested;
    hipError_t launch_err = hipSuccess;
    if (use_gqh_split) {
        const Qwen35DecodeLayerDesc* layers_base =
            static_cast<const Qwen35DecodeLayerDesc*>(layers);
        const Qwen35INT4ScaleDesc* int4_base =
            static_cast<const Qwen35INT4ScaleDesc*>(int4_scales);
        auto phase_grid = [&](int vgpr, size_t scratch, int api_occ, int cap) -> int {
            int waves_s = 1536 / (vgpr > 0 ? vgpr : 256);
            if (scratch > 16 && waves_s > 1) waves_s -= 1;
            if (waves_s < 1) waves_s = 1;
            if (waves_s > 16) waves_s = 16;
            int from_vgpr = (waves_s * 4) / (block_size / 32);
            if (from_vgpr < 1) from_vgpr = 1;
            int g = from_vgpr;
            if (api_occ > 0 && api_occ < g) g = api_occ;
            if (g > cap) g = cap;
            if (g < 1) g = 1;
            return props.multiProcessorCount * g;
        };
        auto launch_split = [&](int split, int layer, int flags, int grid, hipStream_t stream)
            -> hipError_t {
            const Qwen35DecodeLayerDesc* layer_ptr = layers_base + layer;
            const Qwen35INT4ScaleDesc* int4_ptr =
                int4_base != nullptr ? int4_base + layer : nullptr;
            // Split launches pass num_layers=1, so the kernel's
            // batch_descs[0] must be this layer's row, not layer 0.
            const BatchSeqDesc* batch_ptr =
                static_cast<const BatchSeqDesc*>(batch_descs);
            if (batch_ptr != nullptr) {
                batch_ptr += layer;
            }
            if (split == 1) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_persistent_decode_kernel<T, 1>),
                    dim3(static_cast<unsigned int>(grid)),
                    dim3(block_size),
                    shared_bytes,
                    stream,
                    1,
                    hidden_dim,
                    intermediate_size,
                    seqlen_offset,
                    layer_ptr,
                    static_cast<T*>(hidden_io),
                    workspace,
                    counters,
                    barrier_counter,
                    barrier_flag,
                    timing_slots,
                    static_cast<const T*>(cos_table),
                    static_cast<const T*>(sin_table),
                    rotary_dim,
                    proj_buf_floats,
                    attn_scratch_floats,
                    enable_attention_trace,
                    static_cast<const Qwen35FP8ScaleDesc*>(fp8_scales),
                    static_cast<const KVCacheFp8Desc*>(kv_fp8_descs),
                    batch_size,
                    batch_ptr,
                    int4_ptr,
                    flags);
            } else if (split == 2) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_persistent_decode_kernel<T, 2>),
                    dim3(static_cast<unsigned int>(grid)),
                    dim3(block_size),
                    shared_bytes,
                    stream,
                    1,
                    hidden_dim,
                    intermediate_size,
                    seqlen_offset,
                    layer_ptr,
                    static_cast<T*>(hidden_io),
                    workspace,
                    counters,
                    barrier_counter,
                    barrier_flag,
                    timing_slots,
                    static_cast<const T*>(cos_table),
                    static_cast<const T*>(sin_table),
                    rotary_dim,
                    proj_buf_floats,
                    attn_scratch_floats,
                    enable_attention_trace,
                    static_cast<const Qwen35FP8ScaleDesc*>(fp8_scales),
                    static_cast<const KVCacheFp8Desc*>(kv_fp8_descs),
                    batch_size,
                    batch_ptr,
                    int4_ptr,
                    flags);
            } else if (split == 3) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_persistent_decode_kernel<T, 3>),
                    dim3(static_cast<unsigned int>(grid)),
                    dim3(block_size),
                    shared_bytes,
                    stream,
                    1,
                    hidden_dim,
                    intermediate_size,
                    seqlen_offset,
                    layer_ptr,
                    static_cast<T*>(hidden_io),
                    workspace,
                    counters,
                    barrier_counter,
                    barrier_flag,
                    timing_slots,
                    static_cast<const T*>(cos_table),
                    static_cast<const T*>(sin_table),
                    rotary_dim,
                    proj_buf_floats,
                    attn_scratch_floats,
                    enable_attention_trace,
                    static_cast<const Qwen35FP8ScaleDesc*>(fp8_scales),
                    static_cast<const KVCacheFp8Desc*>(kv_fp8_descs),
                    batch_size,
                    batch_ptr,
                    int4_ptr,
                    flags);
            } else if (split == 4) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_persistent_decode_kernel<T, 4>),
                    dim3(static_cast<unsigned int>(grid)),
                    dim3(block_size),
                    shared_bytes,
                    stream,
                    1,
                    hidden_dim,
                    intermediate_size,
                    seqlen_offset,
                    layer_ptr,
                    static_cast<T*>(hidden_io),
                    workspace,
                    counters,
                    barrier_counter,
                    barrier_flag,
                    timing_slots,
                    static_cast<const T*>(cos_table),
                    static_cast<const T*>(sin_table),
                    rotary_dim,
                    proj_buf_floats,
                    attn_scratch_floats,
                    enable_attention_trace,
                    static_cast<const Qwen35FP8ScaleDesc*>(fp8_scales),
                    static_cast<const KVCacheFp8Desc*>(kv_fp8_descs),
                    batch_size,
                    batch_ptr,
                    int4_ptr,
                    flags);
            } else if (split == 5) {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_persistent_decode_kernel<T, 5>),
                    dim3(static_cast<unsigned int>(grid)),
                    dim3(block_size),
                    shared_bytes,
                    stream,
                    1,
                    hidden_dim,
                    intermediate_size,
                    seqlen_offset,
                    layer_ptr,
                    static_cast<T*>(hidden_io),
                    workspace,
                    counters,
                    barrier_counter,
                    barrier_flag,
                    timing_slots,
                    static_cast<const T*>(cos_table),
                    static_cast<const T*>(sin_table),
                    rotary_dim,
                    proj_buf_floats,
                    attn_scratch_floats,
                    enable_attention_trace,
                    static_cast<const Qwen35FP8ScaleDesc*>(fp8_scales),
                    static_cast<const KVCacheFp8Desc*>(kv_fp8_descs),
                    batch_size,
                    batch_ptr,
                    int4_ptr,
                    flags);
            } else {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(supersonic_qwen35_persistent_decode_kernel<T, 6>),
                    dim3(static_cast<unsigned int>(grid)),
                    dim3(block_size),
                    shared_bytes,
                    stream,
                    1,
                    hidden_dim,
                    intermediate_size,
                    seqlen_offset,
                    layer_ptr,
                    static_cast<T*>(hidden_io),
                    workspace,
                    counters,
                    barrier_counter,
                    barrier_flag,
                    timing_slots,
                    static_cast<const T*>(cos_table),
                    static_cast<const T*>(sin_table),
                    rotary_dim,
                    proj_buf_floats,
                    attn_scratch_floats,
                    enable_attention_trace,
                    static_cast<const Qwen35FP8ScaleDesc*>(fp8_scales),
                    static_cast<const KVCacheFp8Desc*>(kv_fp8_descs),
                    batch_size,
                    batch_ptr,
                    int4_ptr,
                    flags);
            }
            return hipGetLastError();
        };
        int occ1 = 0, occ2 = 0, occ3 = 0, occ4 = 0, occ5 = 0, occ6 = 0;
        (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ1,
            (const void*)&supersonic_qwen35_persistent_decode_kernel<T, 1>,
            block_size,
            shared_bytes);
        (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ2,
            (const void*)&supersonic_qwen35_persistent_decode_kernel<T, 2>,
            block_size,
            shared_bytes);
        (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ3,
            (const void*)&supersonic_qwen35_persistent_decode_kernel<T, 3>,
            block_size,
            shared_bytes);
        (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ4,
            (const void*)&supersonic_qwen35_persistent_decode_kernel<T, 4>,
            block_size,
            shared_bytes);
        (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ5,
            (const void*)&supersonic_qwen35_persistent_decode_kernel<T, 5>,
            block_size,
            shared_bytes);
        (void)hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ6,
            (const void*)&supersonic_qwen35_persistent_decode_kernel<T, 6>,
            block_size,
            shared_bytes);
        hipFuncAttributes a1{}, a2{}, a3{}, a4{}, a5{}, a6{};
        (void)hipFuncGetAttributes(
            &a1, (const void*)&supersonic_qwen35_persistent_decode_kernel<T, 1>);
        (void)hipFuncGetAttributes(
            &a2, (const void*)&supersonic_qwen35_persistent_decode_kernel<T, 2>);
        (void)hipFuncGetAttributes(
            &a3, (const void*)&supersonic_qwen35_persistent_decode_kernel<T, 3>);
        (void)hipFuncGetAttributes(
            &a4, (const void*)&supersonic_qwen35_persistent_decode_kernel<T, 4>);
        (void)hipFuncGetAttributes(
            &a5, (const void*)&supersonic_qwen35_persistent_decode_kernel<T, 5>);
        (void)hipFuncGetAttributes(
            &a6, (const void*)&supersonic_qwen35_persistent_decode_kernel<T, 6>);
        static bool dumped_split = false;
        if (!dumped_split) {
            dumped_split = true;
            std::fprintf(
                stderr,
                "[hip-occ] gqh-split in vgpr=%d occ=%d scratch=%zu  "
                "mid vgpr=%d occ=%d scratch=%zu  out vgpr=%d occ=%d scratch=%zu  "
                "gate vgpr=%d occ=%d scratch=%zu  up vgpr=%d occ=%d scratch=%zu  "
                "down vgpr=%d occ=%d scratch=%zu\n",
                a1.numRegs,
                occ1,
                a1.localSizeBytes,
                a2.numRegs,
                occ2,
                a2.localSizeBytes,
                a3.numRegs,
                occ3,
                a3.localSizeBytes,
                a4.numRegs,
                occ4,
                a4.localSizeBytes,
                a6.numRegs,
                occ6,
                a6.localSizeBytes,
                a5.numRegs,
                occ5,
                a5.localSizeBytes);
        }
        GqhMlpHdrs& mlp_hdrs = g_gqh_mlp_hdrs;
        const bool use_gqh_gemv =
            std::getenv("SUPERSONIC_QWEN38_GQH_NOGEMV") == nullptr
            && load_gqh_mlp_hdrs(
                device_ordinal, layers_base, int4_base, num_layers, &mlp_hdrs);
        static bool dumped_gemv = false;
        if (!dumped_gemv) {
            dumped_gemv = true;
            std::fprintf(
                stderr,
                "[hip-occ] gqh-split MLP %s%s\n",
                use_gqh_gemv ? "dedicated GEMV" : "persistent steal",
                use_gqh_gemv ? " + dedicated RMS" : "");
        }
        if (use_gqh_gemv) {
            supersonic_gqh_hip_enable_tight_decode();
            const hipError_t side_err = ensure_decode_side_resources(device_ordinal);
            if (side_err != hipSuccess ||
                decode_side_resources().stream == nullptr ||
                decode_side_resources().events[0] == nullptr ||
                decode_side_resources().events[1] == nullptr) {
                return side_err != hipSuccess ? side_err : hipErrorUnknown;
            }
            float* rms_partials = nullptr;
            const hipError_t rms_err =
                ensure_decode_rms_partials(device_ordinal, &rms_partials);
            if (rms_err != hipSuccess) {
                return rms_err;
            }
            (void)rms_partials;
            auto conv_hdr = [&](const GqhProjHdr& h, int in_d, int out_d) {
                if (h.rung < 0 || h.wire == nullptr || in_d <= 0 || out_d <= 0) {
                    return;
                }
                (void)supersonic_gqh_hip_ensure_tight(
                    device_ordinal,
                    h.rung,
                    const_cast<void*>(h.wire),
                    in_d,
                    out_d);
            };
            bool rec_ready = false;
            for (int li = 0; li < num_layers; ++li) {
                const GqhMixerLayer& mx0 = mlp_hdrs.mix[li];
                if (!rec_ready && mx0.layer_type == 0 && mx0.nv > 0 &&
                    mx0.hkd > 0 && mx0.hvd > 0) {
                    const hipError_t sc_err =
                        ensure_decode_rec_scratch(
                            device_ordinal, mx0.nv, mx0.hkd, mx0.hvd);
                    if (sc_err != hipSuccess) {
                        return 253;
                    }
                    if (ensure_ggml_k_gemv_scratch(device_ordinal, 8192, 8192) !=
                        hipSuccess) {
                        return 252;
                    }
                    rec_ready = true;
                }
                if (mx0.layer_type == 1 && mx0.kv_cache_k != nullptr &&
                    mx0.attn_heads > 0 && mx0.attn_kv_heads > 0 &&
                    mx0.attn_head_dim > 0 && mx0.kv_max_t > 0) {
                    const hipError_t at_err = ensure_decode_full_attn_scratch(
                        device_ordinal,
                        mx0.attn_heads,
                        mx0.attn_kv_heads,
                        mx0.attn_head_dim,
                        mx0.kv_max_t);
                    if (at_err != hipSuccess) {
                        return 251;
                    }
                }
                conv_hdr(mlp_hdrs.gate[li], hidden_dim, intermediate_size);
                conv_hdr(mlp_hdrs.up[li], hidden_dim, intermediate_size);
                conv_hdr(mlp_hdrs.down[li], intermediate_size, hidden_dim);
                if (mx0.layer_type == 1) {
                    conv_hdr(mx0.q, hidden_dim, mx0.q_out);
                    conv_hdr(mx0.k, hidden_dim, mx0.k_out);
                    conv_hdr(mx0.v, hidden_dim, mx0.k_out);
                    conv_hdr(mx0.o, mx0.attn_size, hidden_dim);
                } else {
                    conv_hdr(mx0.qkv, hidden_dim, mx0.qkv_out);
                    conv_hdr(mx0.z, hidden_dim, mx0.z_out);
                    conv_hdr(mx0.lin_out, mx0.val_dim, hidden_dim);
                }
            }
            (void)hipDeviceSynchronize();
            static bool dumped_tight = false;
            if (!dumped_tight) {
                dumped_tight = true;
                std::fprintf(stderr, "[gqh-gemv] planar→tight decode layout enabled\n");
            }
        }
        const int grid_in = use_gqh_gemv
            ? 1
            : phase_grid(a1.numRegs, a1.localSizeBytes, occ1, 4);
        const int grid_mid = use_gqh_gemv
            ? 48
            : phase_grid(a2.numRegs, a2.localSizeBytes, occ2, 6);
        const int grid_out = use_gqh_gemv
            ? 1
            : phase_grid(a3.numRegs, a3.localSizeBytes, occ3, 3);
        const int grid_gate = use_gqh_gemv
            ? 1
            : phase_grid(a4.numRegs, a4.localSizeBytes, occ4, 4);
        const int grid_up = phase_grid(a6.numRegs, a6.localSizeBytes, occ6, 6);
        const int grid_down = use_gqh_gemv
            ? ((hidden_dim + block_size - 1) / block_size)
            : phase_grid(a5.numRegs, a5.localSizeBytes, occ5, 6);
        static bool dumped_grid = false;
        if (!dumped_grid) {
            dumped_grid = true;
            std::fprintf(
                stderr,
                "[hip-occ] gqh-split grids in=%d mid=%d out=%d gate=%d up=%d down=%d\n",
                grid_in,
                grid_mid,
                grid_out,
                grid_gate,
                grid_up,
                grid_down);
        }
        const char* dump_dir =
            std::getenv("SUPERSONIC_QWEN38_DUMP_DECODE_HIDDENS_DIR");
        const char* dump_pos_env =
            std::getenv("SUPERSONIC_QWEN38_DUMP_DECODE_HIDDENS_POS");
        const bool dump_this = dump_dir != nullptr
            && dump_pos_env != nullptr
            && std::atoi(dump_pos_env) == static_cast<int>(seqlen_offset);
        hipDeviceProp_t hip_prop{};
        (void)hipGetDeviceProperties(&hip_prop, device_ordinal);
        // gfx12: Global+side fails; ThreadLocal+side replay was 105 ms/tok
        // vs 61 eager. B>1 single-stream capture on gfx12 was a wash
        // (70 vs 71 ms verify). Keep eager unless gfx11 or opt-in.
        const bool use_graph =
            std::getenv("SUPERSONIC_DISABLE_HIP_GRAPH") == nullptr &&
            std::getenv("SUPERSONIC_DECODE_PROF") == nullptr &&
            (hip_prop.major < 12 ||
             std::getenv("SUPERSONIC_HIP_GRAPH") != nullptr);
        SplitGraphCache& cache = g_split_graph_cache;
        float* ws_hidden = workspace;
        auto dump_ws_f32_n = [&](const char* name, const float* src, int n) {
            if (!dump_this || src == nullptr || n <= 0) {
                return;
            }
            std::vector<float> host(n);
            if (hipMemcpy(
                    host.data(), src, n * sizeof(float), hipMemcpyDeviceToHost)
                != hipSuccess) {
                std::fprintf(stderr, "[dump] D2H %s failed\n", name);
                return;
            }
            char path[768];
            std::snprintf(path, sizeof(path), "%s/%s.f32", dump_dir, name);
            FILE* f = std::fopen(path, "wb");
            if (!f) {
                std::fprintf(stderr, "[dump] open %s failed\n", path);
                return;
            }
            std::fwrite(host.data(), sizeof(float), n, f);
            std::fclose(f);
            const int peek = 3994;
            if (peek < n) {
                std::fprintf(
                    stderr,
                    "[dump] %s n=%d dim3994=%.6f\n",
                    name,
                    n,
                    host[peek]);
            }
        };
        auto dump_ws_f32 = [&](const char* kind, int layer) {
            char name[64];
            std::snprintf(name, sizeof(name), "%s_%02d", kind, layer);
            dump_ws_f32_n(name, ws_hidden, hidden_dim);
        };
        const char* dump_lin_env =
            std::getenv("SUPERSONIC_QWEN38_DUMP_LINEAR_LAYER");
        const int dump_lin =
            (dump_this && dump_lin_env) ? std::atoi(dump_lin_env) : -1;
        float* ws_normed = ws_hidden + batch_size * hidden_dim;
        float* ws_gate = ws_normed + batch_size * hidden_dim;
        float* ws_up = ws_gate + intermediate_size;
        float* ws_mlp = ws_gate + batch_size * intermediate_size * 2;
        float* ws_token = ws_mlp + batch_size * hidden_dim;
        float* ws_proj = ws_token + batch_size * hidden_dim;
        float* ws_attn = ws_proj + batch_size * proj_buf_floats;
        const int gemv_ncols = batch_size > 0 ? batch_size : 1;
        if (gemv_ncols > 1) {
            const hipError_t mtp_err =
                ensure_mtp_prefix_snap(mlp_hdrs, num_layers, gemv_ncols);
            if (mtp_err != hipSuccess) {
                return mtp_err;
            }
        } else {
            mtp_prefix_snap().ready = false;
        }
        const int64_t hidden_stride = hidden_dim;
        const int64_t proj_stride = proj_buf_floats;
        const int64_t mlp_stride = static_cast<int64_t>(intermediate_size) * 2;
        const int64_t attn_stride = attn_scratch_floats;
        const char* out_mode_env = std::getenv("SUPERSONIC_QWEN38_GQH_GEMV_OUT");
        int host_out_mask = 0;
        if (use_gqh_gemv) {
            if (out_mode_env == nullptr || std::strcmp(out_mode_env, "all") == 0) {
                host_out_mask = 3;
            } else if (std::strcmp(out_mode_env, "lin") == 0) {
                host_out_mask = 1;
            } else if (std::strcmp(out_mode_env, "full") == 0) {
                host_out_mask = 2;
            } else if (std::strcmp(out_mode_env, "0") == 0 ||
                       std::strcmp(out_mode_env, "off") == 0) {
                host_out_mask = 0;
            } else {
                host_out_mask = 3;
            }
        }
        auto record_layers = [&](hipStream_t stream) -> hipError_t {
            hipError_t err = hipSuccess;
            static int dec_prof_step = 0;
            static const bool dec_prof = [] {
                const char* e = std::getenv("SUPERSONIC_DECODE_PROF");
                return e != nullptr && e[0] != '\0' && e[0] != '0';
            }();
            double ms_attn = 0, ms_inproj = 0, ms_rec = 0, ms_out = 0, ms_mlp = 0;
            double ms_pair = 0, ms_down = 0, ms_rms = 0, ms_prep = 0;
            auto now_ms = []() {
                return std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count();
            };
            auto sync_now = [&]() {
                if (stream) {
                    (void)hipStreamSynchronize(stream);
                } else {
                    (void)hipDeviceSynchronize();
                }
            };
            for (int layer = 0; layer < num_layers; ++layer) {
                int in_flags = (layer == 0) ? 1 : 0;
                int mlp_flags = (layer == num_layers - 1) ? 2 : 0;
                if (use_gqh_gemv) {
                    const GqhMixerLayer& mx = mlp_hdrs.mix[layer];
                    if (mx.layer_type == 1) {
                        if (proj_can_gemv(mx.k)) {
                            in_flags |= 8;
                        }
                        if (proj_can_gemv(mx.v)) {
                            in_flags |= 16;
                        }
                    } else {
                        if (proj_can_gemv(mx.b)) {
                            in_flags |= 8;
                        }
                        if (proj_can_gemv(mx.a)) {
                            in_flags |= 16;
                        }
                    }
                    in_flags |= 32;
                }
                DecodeSideJoinGuard z_pending(device_ordinal, stream);
                if (use_gqh_gemv) {
                    const GqhMixerLayer& mx_rms = mlp_hdrs.mix[layer];
                    int rms_flags = 2;
                    const hip_bfloat16* io = nullptr;
                    if ((in_flags & 1) != 0) {
                        rms_flags |= 1;
                        io = static_cast<const hip_bfloat16*>(hidden_io);
                    }
                    const double t_rms0 = dec_prof ? (sync_now(), now_ms()) : 0;
                    err = launch_decode_rms(
                        ws_hidden,
                        ws_normed,
                        io,
                        static_cast<const hip_bfloat16*>(mx_rms.input_norm_w),
                        hidden_dim,
                        batch_size,
                        mx_rms.input_norm_eps,
                        mx_rms.rms_unit_offset ? 1.0f : 0.0f,
                        rms_flags,
                        device_ordinal,
                        stream);
                    if (dec_prof) {
                        sync_now();
                        ms_rms += now_ms() - t_rms0;
                    }
                } else {
                    err = launch_split(1, layer, in_flags, grid_in, stream);
                }
                if (err != hipSuccess) {
                    std::fprintf(
                        stderr,
                        "[decode] fail after in-rms layer=%d: %s\n",
                        layer,
                        hipGetErrorString(err));
                    return err;
                }
                if (layer == dump_lin) {
                    if (stream) {
                        (void)hipStreamSynchronize(stream);
                    } else {
                        (void)hipDeviceSynchronize();
                    }
                    dump_ws_f32_n("l5_in", ws_hidden, hidden_dim);
                    dump_ws_f32_n("l5_normed", ws_normed, hidden_dim);
                }
                if (use_gqh_gemv) {
                    const GqhMixerLayer& mx = mlp_hdrs.mix[layer];
                    const double t_in0 = dec_prof ? (sync_now(), now_ms()) : 0;
                    if (mx.layer_type == 1) {
                        hipStream_t kv_stream = stream;
                        hipStream_t side = nullptr;
                        if (stream == nullptr) {
                            err = decode_side_stream(device_ordinal, &side);
                            if (err != hipSuccess) {
                                return err;
                            }
                        }
                        if (side != nullptr && mx.k.wire != nullptr) {
                            err = decode_fork_side(device_ordinal, stream, side);
                            if (err != hipSuccess) {
                                return err;
                            }
                            kv_stream = side;
                        }
                        err = launch_mixer_proj(
                            device_ordinal, mx.q, ws_normed, ws_proj,
                            hidden_dim, mx.q_out, stream, false, gemv_ncols,
                            hidden_stride, proj_stride);
                        if (err == hipSuccess) {
                            err = launch_mixer_proj(
                                device_ordinal, mx.k, ws_normed,
                                ws_proj + mx.q_out,
                                hidden_dim, mx.k_out, kv_stream, false,
                                gemv_ncols, hidden_stride, proj_stride);
                        }
                        if (err == hipSuccess) {
                            err = launch_mixer_proj(
                                device_ordinal, mx.v, ws_normed,
                                ws_proj + mx.q_out + mx.k_out,
                                hidden_dim, mx.k_out, kv_stream, false,
                                gemv_ncols, hidden_stride, proj_stride);
                        }
                        if (kv_stream != stream) {
                            err = decode_join_side(device_ordinal, stream, kv_stream);
                            if (err != hipSuccess) {
                                return err;
                            }
                        }
                    } else {
                        hipStream_t z_stream = stream;
                        hipStream_t side = nullptr;
                        if (stream == nullptr) {
                            err = decode_side_stream(device_ordinal, &side);
                            if (err != hipSuccess) {
                                return err;
                            }
                        }
                        if (side != nullptr && mx.z.wire != nullptr &&
                            mx.z_out > 0) {
                            err = decode_fork_side(device_ordinal, stream, side);
                            if (err != hipSuccess) {
                                return err;
                            }
                            z_stream = side;
                        }
                        err = launch_mixer_proj(
                            device_ordinal, mx.qkv, ws_normed, ws_proj,
                            hidden_dim, mx.qkv_out, stream, false, gemv_ncols,
                            hidden_stride, proj_stride);
                        if (err == hipSuccess) {
                            err = launch_mixer_proj(
                                device_ordinal, mx.z, ws_normed,
                                ws_proj + mx.qkv_out,
                                hidden_dim, mx.z_out, z_stream, false,
                                gemv_ncols, hidden_stride, proj_stride);
                        }
                        // Join z after rec-prep: rec-prep only reads qkv, so
                        // it can overlap the z GEMV. Rec fused still needs z.
                        if (z_stream != stream) {
                            z_pending.defer(z_stream);
                        }
                        if (err == hipSuccess && mx.b.qtype != 8) {
                            err = launch_mixer_proj(
                                device_ordinal, mx.b, ws_normed,
                                ws_proj + mx.qkv_out + mx.z_out,
                                hidden_dim, mx.nv, stream, false, gemv_ncols,
                                hidden_stride, proj_stride);
                        }
                        if (err == hipSuccess && mx.a.qtype != 8) {
                            err = launch_mixer_proj(
                                device_ordinal, mx.a, ws_normed,
                                ws_proj + mx.qkv_out + mx.z_out + mx.nv,
                                hidden_dim, mx.nv, stream, false, gemv_ncols,
                                hidden_stride, proj_stride);
                        }
                    }

                    if (dec_prof) {
                        sync_now();
                        ms_inproj += now_ms() - t_in0;
                    }
                    if (err != hipSuccess) {
                        std::fprintf(
                            stderr,
                            "[decode] fail after inproj layer=%d: %s\n",
                            layer,
                            hipGetErrorString(err));
                        return err;
                    }
                    if (layer == dump_lin) {
                        if (z_pending.active()) {
                            err = z_pending.join();
                            if (err != hipSuccess) {
                                return err;
                            }
                        }
                        if (stream) {
                            (void)hipStreamSynchronize(stream);
                        } else {
                            (void)hipDeviceSynchronize();
                        }
                        const GqhMixerLayer& mxd = mlp_hdrs.mix[layer];
                        if (mxd.layer_type == 1) {
                            dump_ws_f32_n("l5_q", ws_proj, mxd.q_out);
                            dump_ws_f32_n("l5_k", ws_proj + mxd.q_out, mxd.k_out);
                            dump_ws_f32_n(
                                "l5_v",
                                ws_proj + mxd.q_out + mxd.k_out,
                                mxd.k_out);
                        } else {
                            dump_ws_f32_n("l5_qkv", ws_proj, mxd.qkv_out);
                            dump_ws_f32_n("l5_z", ws_proj + mxd.qkv_out, mxd.z_out);
                        }
                    }
                }
                int mid_flags = 0;
                int mid_g = grid_mid;
                bool rec_prep = false;
                if (use_gqh_gemv) {
                    const GqhMixerLayer& mx = mlp_hdrs.mix[layer];
                    if (mx.layer_type == 1) {
                        const int nh = mx.attn_heads > 0 ? mx.attn_heads : 1;
                        // 24-block mid wins once KV is long; at Hello-length
                        // the grid_barrier tax is larger than the head parallel.
                        // Batched verify (B>1) has enough QK-norm/RoPE/KV
                        // work to keep the 24-wide grid busy.
                        mid_g = (seqlen_offset < 24 && gemv_ncols <= 1) ? 1 : nh;
                        if (mx.kv_cache_k != nullptr && mx.attn_head_dim > 0 &&
                            mx.attn_kv_heads > 0) {
                            mid_flags |= 256;
                        }
                    } else {
                        if (proj_can_gemv(mx.b) && mx.b.qtype == 8) {
                            mid_flags |= 8;
                        }
                        if (proj_can_gemv(mx.a) && mx.a.qtype == 8) {
                            mid_flags |= 16;
                        }
                        if (mx.recurrent_state != nullptr && mx.hkd > 0 &&
                            mx.hvd == mx.hkd && mx.hkd * 2 <= 256 &&
                            mx.nv > 0 && mx.qkv_out > mx.nv * mx.hvd) {
                            mid_flags |= 128;
                            rec_prep = mx.conv_state != nullptr &&
                                mx.conv1d_w != nullptr &&
                                mx.conv_kernel_size == 4 &&
                                mx.hkd == 128;
                        }
                    }
                }
                if (layer == dump_lin) {
                    Qwen35DecodeLayerDesc Ld{};
                    if (hipMemcpy(
                            &Ld,
                            static_cast<const Qwen35DecodeLayerDesc*>(layers) +
                                layer,
                            sizeof(Ld),
                            hipMemcpyDeviceToHost) == hipSuccess &&
                        Ld.recurrent_state != nullptr) {
                        dump_ws_f32_n(
                            "l5_rec_in",
                            static_cast<const float*>(Ld.recurrent_state),
                            Ld.linear_num_v_heads * Ld.linear_head_k_dim *
                                Ld.linear_head_v_dim);
                    }
                }
                if (rec_prep) {
                    const double t_p0 = dec_prof ? (sync_now(), now_ms()) : 0;
                    const GqhMixerLayer& mxp = mlp_hdrs.mix[layer];
                    const int nk_p =
                        (mxp.qkv_out - mxp.nv * mxp.hvd) / (2 * mxp.hkd);
                    // ggml-K matvec (12×128) has far more occupancy than
                    // WMMA m=1 tile16 (3×32) on nv=48. B=1 Hello tokens
                    // must stay 9419…3242; revert if they move.
                    if ((mid_flags & 8) != 0) {
                        (void)launch_ggml_k_gemv(
                            mxp.b,
                            ws_normed,
                            ws_proj + mxp.qkv_out + mxp.z_out,
                            hidden_dim,
                            mxp.nv,
                            stream,
                            gemv_ncols,
                            hidden_stride,
                            proj_stride);
                    }
                    if ((mid_flags & 16) != 0) {
                        (void)launch_ggml_k_gemv(
                            mxp.a,
                            ws_normed,
                            ws_proj + mxp.qkv_out + mxp.z_out + mxp.nv,
                            hidden_dim,
                            mxp.nv,
                            stream,
                            gemv_ncols,
                            hidden_stride,
                            proj_stride);
                    }
                    {
                        MtpPrefixSnap& psnap = mtp_prefix_snap();
                        hip_bfloat16* conv_snaps = nullptr;
                        int64_t snap_stride = 0;
                        if (psnap.ready && psnap.conv[0][layer] != nullptr &&
                            psnap.conv_bytes[layer] > 0) {
                            conv_snaps = psnap.conv[0][layer];
                            snap_stride = static_cast<int64_t>(
                                psnap.conv_bytes[layer] / sizeof(hip_bfloat16));
                        }
                        launch_decode_rec_prep(
                            mxp.qkv_out,
                            nk_p,
                            mxp.hkd,
                            ws_proj,
                            static_cast<hip_bfloat16*>(mxp.conv_state),
                            static_cast<const hip_bfloat16*>(mxp.conv1d_w),
                            ws_gate,
                            stream,
                            gemv_ncols,
                            proj_stride,
                            mlp_stride,
                            conv_snaps,
                            snap_stride);
                    }
                    if (z_pending.active()) {
                        err = z_pending.join();
                        if (err != hipSuccess) {
                            return err;
                        }
                    }
                    if (dec_prof) {
                        sync_now();
                        ms_prep += now_ms() - t_p0;
                    }
                } else {
                    if (z_pending.active()) {
                        err = z_pending.join();
                        if (err != hipSuccess) {
                            return err;
                        }
                    }
                    if ((mid_flags & 256) != 0) {
                        err = launch_decode_full_prep(
                            mlp_hdrs.mix[layer],
                            ws_proj,
                            ws_attn,
                            cos_table,
                            sin_table,
                            seqlen_offset,
                            gemv_ncols,
                            proj_buf_floats,
                            attn_scratch_floats,
                            stream);
                    } else {
                        err = launch_split(2, layer, mid_flags, mid_g, stream);
                    }
                    if (err != hipSuccess) return err;
                }
                if ((mid_flags & 256) != 0) {
                    const GqhMixerLayer& mxf = mlp_hdrs.mix[layer];
                    float* saved_gate =
                        ws_attn + mxf.attn_heads * mxf.attn_head_dim;
                    const double t_a0 = dec_prof ? (sync_now(), now_ms()) : 0;
                    err = launch_host_full_attn(
                        device_ordinal,
                        mxf,
                        ws_proj,
                        saved_gate,
                        seqlen_offset,
                        stream,
                        gemv_ncols,
                        proj_buf_floats,
                        attn_scratch_floats);
                    if (dec_prof) {
                        sync_now();
                        ms_attn += now_ms() - t_a0;
                    }
                    if (err != hipSuccess) return err;
                }
                if ((mid_flags & 128) != 0) {
                    const GqhMixerLayer& mxr = mlp_hdrs.mix[layer];
                    const int nv = mxr.nv;
                    const int hkd = mxr.hkd;
                    const int hvd = mxr.hvd;
                    const int nk =
                        (mxr.qkv_out - nv * hvd) / (2 * hkd);
                    const int key_dim = nk * hkd;
                    DecodeRecScratch& sc = decode_rec_scratch();
                    const double t_r0 = dec_prof ? (sync_now(), now_ms()) : 0;
                    for (int b = 0; b < gemv_ncols; ++b) {
                        float* conv_b =
                            ws_gate + static_cast<int64_t>(b) * mlp_stride;
                        float* proj_b =
                            ws_proj + static_cast<int64_t>(b) * proj_stride;
                        float* attn_b =
                            ws_attn + static_cast<int64_t>(b) * attn_stride;
                        const float* q_u = conv_b + mxr.qkv_out;
                        const float* k_u = conv_b + mxr.qkv_out + key_dim;
                        const float* v_ptr = conv_b + 2 * key_dim;
                        const float* b_ptr =
                            proj_b + mxr.qkv_out + mxr.z_out;
                        const float* a_ptr = b_ptr + nv;
                        const float* z_ptr = proj_b + mxr.qkv_out;
                        if (hkd == 128 && hvd == 128) {
                            const int st =
                                supersonic_qwen35_hip_decode_rec_k128_fused(
                                    device_ordinal,
                                    nv,
                                    nk,
                                    static_cast<float*>(mxr.recurrent_state),
                                    q_u,
                                    k_u,
                                    v_ptr,
                                    b_ptr,
                                    a_ptr,
                                    static_cast<const hip_bfloat16*>(
                                        mxr.dt_bias_w),
                                    static_cast<const hip_bfloat16*>(
                                        mxr.a_log_exp_w),
                                    sc.out,
                                    stream);
                            err = st == 0 ? hipSuccess : hipErrorInvalidValue;
                        } else {
                            launch_decode_pack_rec_inputs(
                                nv,
                                nk,
                                hkd,
                                q_u,
                                k_u,
                                b_ptr,
                                a_ptr,
                                static_cast<const hip_bfloat16*>(mxr.dt_bias_w),
                                static_cast<const hip_bfloat16*>(
                                    mxr.a_log_exp_w),
                                sc.q,
                                sc.k,
                                sc.beta,
                                sc.g,
                                stream);
                            err = launch_delta_recurrent_prefill_f32(
                                device_ordinal,
                                nv,
                                hkd,
                                hvd,
                                static_cast<const float*>(mxr.recurrent_state),
                                sc.q,
                                sc.k,
                                v_ptr,
                                sc.beta,
                                sc.g,
                                sc.out,
                                stream);
                        }
                        if (err != hipSuccess) return err;
                        launch_decode_extract_rec_gated(
                            nv,
                            hkd,
                            hvd,
                            mxr.linear_norm_eps,
                            sc.out,
                            z_ptr,
                            static_cast<const float*>(mxr.linear_norm_w),
                            (hkd == 128 && hvd == 128)
                                ? nullptr
                                : static_cast<float*>(mxr.recurrent_state),
                            attn_b,
                            stream,
                            (hkd == 128 && hvd == 128) ? 1 : 0);
                        MtpPrefixSnap& psnap = mtp_prefix_snap();
                        if (psnap.ready && b < gemv_ncols &&
                            psnap.rec[b][layer] != nullptr &&
                            psnap.rec_bytes[layer] > 0 &&
                            mxr.recurrent_state != nullptr) {
                            (void)hipMemcpyAsync(
                                psnap.rec[b][layer],
                                mxr.recurrent_state,
                                psnap.rec_bytes[layer],
                                hipMemcpyDeviceToDevice,
                                stream);
                        }
                    }
                    if (dec_prof) {
                        sync_now();
                        ms_rec += now_ms() - t_r0;
                    }
                    static bool dumped_host_rec = false;
                    if (!dumped_host_rec) {
                        dumped_host_rec = true;
                        std::fprintf(
                            stderr,
                            "[decode] host delta_recurrent_prefill nv=%d "
                            "nk=%d khd=%d\n",
                            nv,
                            nk,
                            hkd);
                    }
                }
                if (layer == dump_lin) {
                    if (stream) {
                        (void)hipStreamSynchronize(stream);
                    } else {
                        (void)hipDeviceSynchronize();
                    }
                    const GqhMixerLayer& mxd = mlp_hdrs.mix[layer];
                    if (mxd.layer_type == 1) {
                        dump_ws_f32_n("l5_attn", ws_proj, mxd.attn_size);
                        if ((mid_flags & 256) != 0) {
                            DecodeFullAttnScratch& scq = decode_full_attn_scratch();
                            const int qn = mxd.attn_heads * mxd.attn_head_dim;
                            std::vector<float> qh(static_cast<size_t>(qn));
                            std::vector<hip_bfloat16> qb(static_cast<size_t>(qn));
                            if (scq.q != nullptr &&
                                hipMemcpy(
                                    qb.data(),
                                    scq.q,
                                    static_cast<size_t>(qn) * sizeof(hip_bfloat16),
                                    hipMemcpyDeviceToHost) == hipSuccess) {
                                for (int i = 0; i < qn; ++i) {
                                    qh[i] = static_cast<float>(qb[i]);
                                }
                                char path[768];
                                std::snprintf(
                                    path, sizeof(path), "%s/l5_qrope.f32", dump_dir);
                                FILE* f = std::fopen(path, "wb");
                                if (f) {
                                    std::fwrite(
                                        qh.data(), sizeof(float), qh.size(), f);
                                    std::fclose(f);
                                }
                            }
                        }
                    }
                    dump_ws_f32_n("l5_conv", ws_gate, mxd.qkv_out);
                    dump_ws_f32_n(
                        "l5_b",
                        ws_proj + mxd.qkv_out + mxd.z_out,
                        mxd.nv);
                    dump_ws_f32_n(
                        "l5_a",
                        ws_proj + mxd.qkv_out + mxd.z_out + mxd.nv,
                        mxd.nv);
                    dump_ws_f32_n("l5_qnorm", ws_gate + mxd.qkv_out, mxd.qkv_out / 5);
                    dump_ws_f32_n(
                        "l5_knorm",
                        ws_gate + mxd.qkv_out + mxd.qkv_out / 5,
                        mxd.qkv_out / 5);
                    if ((mid_flags & 128) != 0 && mxd.hkd > 0 &&
                        mxd.hvd > 0 && mxd.nv > 0) {
                        DecodeRecScratch& scd = decode_rec_scratch();
                        std::vector<float> rec(
                            static_cast<size_t>(mxd.nv) *
                            static_cast<size_t>(mxd.hvd));
                        const bool compact_rec =
                            mxd.hkd == 128 && mxd.hvd == 128;
                        for (int h = 0; h < mxd.nv; ++h) {
                            const size_t src_off = compact_rec
                                ? static_cast<size_t>(h) * mxd.hvd
                                : static_cast<size_t>(h) *
                                    (1 + mxd.hkd) * mxd.hvd;
                            hipMemcpy(
                                rec.data() +
                                    static_cast<size_t>(h) * mxd.hvd,
                                scd.out + src_off,
                                static_cast<size_t>(mxd.hvd) *
                                    sizeof(float),
                                hipMemcpyDeviceToHost);
                        }
                        char path[768];
                        std::snprintf(
                            path, sizeof(path), "%s/l5_rec.f32", dump_dir);
                        FILE* f = std::fopen(path, "wb");
                        if (f) {
                            std::fwrite(
                                rec.data(),
                                sizeof(float),
                                rec.size(),
                                f);
                            std::fclose(f);
                        }
                    } else {
                        dump_ws_f32_n(
                            "l5_rec",
                            ws_gate + mxd.qkv_out + 2 * (mxd.qkv_out / 5),
                            mxd.val_dim);
                    }
                    dump_ws_f32_n("l5_gated", ws_attn, mxd.val_dim);
                    dump_ws_f32_n("l5_mid_hidden", ws_hidden, hidden_dim);
                }
                int out_flags = 0;
                if (use_gqh_gemv) {
                    const GqhMixerLayer& mx = mlp_hdrs.mix[layer];
                    const bool do_full = mx.layer_type == 1 &&
                        (host_out_mask & 2) != 0 && proj_can_gemv(mx.o);
                    const bool do_lin = mx.layer_type == 0 &&
                        (host_out_mask & 1) != 0 && proj_can_gemv(mx.lin_out);
                    const double t_o0 = dec_prof ? (sync_now(), now_ms()) : 0;
                    if (do_full) {
                        err = launch_mixer_proj_acc(
                            device_ordinal, mx.o, ws_proj, ws_hidden,
                            mx.attn_size, hidden_dim, stream, gemv_ncols,
                            proj_stride, hidden_stride);
                    } else if (do_lin) {
                        // extract_rec_gated already BF16-rounds attn_out,
                        // so a second round_f32 is a no-op. Acc GEMV is
                        // y = round(y + round(acc)), same as GEMV + add_round
                        // after the extra y-round collapses.
                        err = launch_mixer_proj_acc(
                            device_ordinal, mx.lin_out, ws_attn, ws_hidden,
                            mx.val_dim, hidden_dim, stream, gemv_ncols,
                            attn_stride, hidden_stride);
                    }
                    if (dec_prof) {
                        sync_now();
                        ms_out += now_ms() - t_o0;
                    }
                    if (err != hipSuccess) return err;
                    if (do_full || do_lin) {
                        static bool dumped_out_cmp = false;
                        if (!dumped_out_cmp &&
                            std::getenv("SUPERSONIC_QWEN38_GQH_GEMV_OUT_CMP") !=
                                nullptr) {
                            dumped_out_cmp = true;
                            const int in_dim = do_full ? mx.attn_size : mx.val_dim;
                            const float* xptr = do_full ? ws_proj : ws_attn;
                            float* saved = ws_mlp;
                            float* fused = do_lin ? (ws_attn + mx.val_dim) : ws_attn;
                            hipMemcpyAsync(
                                saved,
                                ws_hidden,
                                static_cast<size_t>(hidden_dim) * sizeof(float),
                                hipMemcpyDeviceToDevice,
                                stream);
                            hipMemcpyAsync(
                                fused,
                                ws_hidden,
                                static_cast<size_t>(hidden_dim) * sizeof(float),
                                hipMemcpyDeviceToDevice,
                                stream);
                            launch_add_round_f32(fused, ws_token, hidden_dim, stream);
                            err = launch_split(3, layer, 0, grid_out, stream);
                            if (err != hipSuccess) return err;
                            hipError_t sync_err = hipStreamSynchronize(stream);
                            if (sync_err != hipSuccess) {
                                return sync_err;
                            }
                            float hx[4] = {}, hy[4] = {}, hf[4] = {}, hs[4] = {};
                            hipMemcpy(hx, xptr, sizeof(hx), hipMemcpyDeviceToHost);
                            hipMemcpy(hy, ws_token, sizeof(hy), hipMemcpyDeviceToHost);
                            hipMemcpy(hf, fused, sizeof(hf), hipMemcpyDeviceToHost);
                            hipMemcpy(hs, ws_hidden, sizeof(hs), hipMemcpyDeviceToHost);
                            float max_abs = 0.0f;
                            int n_diff = 0;
                            const int chunk = 256;
                            float host_f[256];
                            float host_s[256];
                            for (int off = 0; off < hidden_dim; off += chunk) {
                                const int n = hidden_dim - off < chunk
                                    ? hidden_dim - off
                                    : chunk;
                                hipMemcpy(
                                    host_f,
                                    fused + off,
                                    static_cast<size_t>(n) * sizeof(float),
                                    hipMemcpyDeviceToHost);
                                hipMemcpy(
                                    host_s,
                                    ws_hidden + off,
                                    static_cast<size_t>(n) * sizeof(float),
                                    hipMemcpyDeviceToHost);
                                for (int i = 0; i < n; ++i) {
                                    const float d = host_f[i] - host_s[i];
                                    const float ad = d < 0 ? -d : d;
                                    if (ad > max_abs) {
                                        max_abs = ad;
                                    }
                                    if (ad > 1e-6f) {
                                        ++n_diff;
                                    }
                                }
                            }
                            std::fprintf(
                                stderr,
                                "[gqh-gemv] CMP L%d type=%d in=%d out=%d "
                                "rung=%d sc=%.6g grid=%d x0=%.6g,%.6g,%.6g,%.6g "
                                "y0=%.6g,%.6g,%.6g,%.6g fused0=%.6g steal0=%.6g "
                                "max_abs=%.6g n_diff=%d/%d\n",
                                layer,
                                mx.layer_type,
                                in_dim,
                                hidden_dim,
                                do_full ? mx.o.rung : mx.lin_out.rung,
                                do_full ? mx.o.scale : mx.lin_out.scale,
                                do_full ? mx.o.grid : mx.lin_out.grid,
                                hx[0],
                                hx[1],
                                hx[2],
                                hx[3],
                                hy[0],
                                hy[1],
                                hy[2],
                                hy[3],
                                hf[0],
                                hs[0],
                                max_abs,
                                n_diff,
                                hidden_dim);
                            (void)saved;
                        } else {
                            out_flags = 4;
                        }
                    }
                }
                // out_flags==4: GEMV already wrote the residual. Split-3
                // would only grid-barrier and re-checkpoint hidden_f32.
                if (out_flags != 4) {
                    err = launch_split(3, layer, out_flags, grid_out, stream);
                    if (err != hipSuccess) return err;
                }
                if (dump_this) {
                    if (stream) {
                        (void)hipStreamSynchronize(stream);
                    } else {
                        (void)hipDeviceSynchronize();
                    }
                    dump_ws_f32("attn", layer);
                }
                if (use_gqh_gemv) {
                    const GqhMixerLayer& mx_rms = mlp_hdrs.mix[layer];
                    const double t_rms1 = dec_prof ? (sync_now(), now_ms()) : 0;
                    err = launch_decode_rms(
                        ws_hidden,
                        ws_normed,
                        nullptr,
                        static_cast<const hip_bfloat16*>(mx_rms.post_attn_norm_w),
                        hidden_dim,
                        batch_size,
                        mx_rms.post_attn_norm_eps,
                        mx_rms.rms_unit_offset ? 1.0f : 0.0f,
                        0,
                        device_ordinal,
                        stream);
                    if (dec_prof) {
                        sync_now();
                        ms_rms += now_ms() - t_rms1;
                    }
                } else {
                    err = launch_split(4, layer, 0, grid_gate, stream);
                }
                if (err != hipSuccess) return err;
                if (use_gqh_gemv) {
                    const double t_m0 = dec_prof ? (sync_now(), now_ms()) : 0;
                    const GqhProjHdr& gate_h = mlp_hdrs.gate[layer];
                    const GqhProjHdr& up_h = mlp_hdrs.up[layer];
                    if (gate_h.rung >= 0 && up_h.rung >= 0 &&
                        gate_h.rung == up_h.rung) {
                        err = launch_gqh_gemv_pair(
                            device_ordinal,
                            gate_h,
                            up_h,
                            ws_normed,
                            ws_gate,
                            ws_up,
                            hidden_dim,
                            intermediate_size,
                            intermediate_size,
                            stream,
                            true,
                            gemv_ncols,
                            hidden_stride,
                            mlp_stride);
                    } else {
                        err = hipErrorInvalidValue;
                    }
                    if (err != hipSuccess) {
                        err = launch_gqh_gemv(
                            device_ordinal,
                            gate_h,
                            ws_normed,
                            ws_gate,
                            hidden_dim,
                            intermediate_size,
                            stream,
                            gemv_ncols,
                            hidden_stride,
                            mlp_stride);
                        if (err == hipSuccess) {
                            err = launch_gqh_gemv(
                                device_ordinal,
                                up_h,
                                ws_normed,
                                ws_up,
                                hidden_dim,
                                intermediate_size,
                                stream,
                                gemv_ncols,
                                hidden_stride,
                                mlp_stride);
                        }
                        if (err == hipSuccess) {
                            launch_swiglu_f32(
                                ws_gate, ws_up, intermediate_size, stream,
                                gemv_ncols, mlp_stride);
                        }
                    }
                    if (err != hipSuccess) {
                        std::fprintf(
                            stderr,
                            "[decode] fail after pair layer=%d: %s\n",
                            layer,
                            hipGetErrorString(err));
                        return err;
                    }
                    const double t_d0 = dec_prof ? (sync_now(), now_ms()) : 0;
                    if (dec_prof) {
                        ms_pair += t_d0 - t_m0;
                    }
                    if (mlp_hdrs.down[layer].rung >= 0) {
                        err = launch_gqh_gemv_acc(
                            device_ordinal,
                            mlp_hdrs.down[layer],
                            ws_gate,
                            ws_hidden,
                            intermediate_size,
                            hidden_dim,
                            stream,
                            gemv_ncols,
                            mlp_stride,
                            hidden_stride);
                    } else {
                        err = launch_mixer_proj_acc(
                            device_ordinal,
                            mlp_hdrs.down[layer],
                            ws_gate,
                            ws_hidden,
                            intermediate_size,
                            hidden_dim,
                            stream,
                            gemv_ncols,
                            mlp_stride,
                            hidden_stride);
                    }
                    if (err != hipSuccess) {
                        std::fprintf(
                            stderr,
                            "[decode] fail after down layer=%d: %s\n",
                            layer,
                            hipGetErrorString(err));
                        return err;
                    }
                    if (dec_prof) {
                        sync_now();
                        const double t_m1 = now_ms();
                        ms_down += t_m1 - t_d0;
                        ms_mlp += t_m1 - t_m0;
                    }
                    if ((mlp_flags & 2) != 0) {
                        err = launch_hidden_store_bf16(
                            ws_hidden,
                            static_cast<hip_bfloat16*>(hidden_io),
                            batch_size * hidden_dim,
                            stream);
                        if (err != hipSuccess) return err;
                    }
                } else {
                    err = launch_split(6, layer, 0, grid_up, stream);
                    if (err != hipSuccess) return err;
                    err = launch_split(5, layer, mlp_flags, grid_down, stream);
                    if (err != hipSuccess) return err;
                }
                if (dump_this) {
                    if (stream) {
                        (void)hipStreamSynchronize(stream);
                    } else {
                        (void)hipDeviceSynchronize();
                    }
                    dump_ws_f32("hidden", layer);
                }
            }
            if (dump_this) {
                std::fprintf(
                    stderr,
                    "[dump] megakernel decode hiddens pos=%d layers=%d dir=%s\n",
                    static_cast<int>(seqlen_offset),
                    num_layers,
                    dump_dir);
            }
            if (dec_prof) {
                std::fprintf(
                    stderr,
                    "[decode-prof] step=%d seq=%d attn=%.1f inproj=%.1f rec=%.1f "
                    "out=%.1f mlp=%.1f pair=%.1f down=%.1f rms=%.1f prep=%.1f ms\n",
                    dec_prof_step,
                    static_cast<int>(seqlen_offset),
                    ms_attn,
                    ms_inproj,
                    ms_rec,
                    ms_out,
                    ms_mlp,
                    ms_pair,
                    ms_down,
                    ms_rms,
                    ms_prep);
                ++dec_prof_step;
            }
            return hipSuccess;
        };
        launch_err = hipMemcpyToSymbol(
            HIP_SYMBOL(qwen35_gemv_out_mask),
            &host_out_mask,
            sizeof(host_out_mask));
        if (launch_err != hipSuccess) {
            return 254;
        }
        const int host_seqlen = seqlen_offset;
        launch_err = hipMemcpyToSymbol(
            HIP_SYMBOL(qwen35_split_seqlen), &host_seqlen, sizeof(host_seqlen));
        if (launch_err != hipSuccess) {
            // fall through to eager launches
        } else if (use_graph) {
            const bool reuse = cache.exec != nullptr
                && cache.num_layers == num_layers
                && cache.device_ordinal == device_ordinal
                && cache.grid_in == grid_in
                && cache.grid_mid == grid_mid
                && cache.grid_out == grid_out
                && cache.grid_gate == grid_gate
                && cache.grid_up == grid_up
                && cache.grid_down == grid_down
                && cache.layers == layers
                && cache.hidden_io == hidden_io
                && cache.workspace == workspace
                && cache.counters == counters
                && cache.barrier_counter == barrier_counter
                && cache.barrier_flag == barrier_flag
                && cache.int4 == int4_scales
                && cache.cos_table == cos_table
                && cache.sin_table == sin_table
                && cache.fp8_scales == fp8_scales
                && cache.kv_fp8_descs == kv_fp8_descs
                && cache.batch_descs == batch_descs
                && cache.state_signature ==
                    (use_gqh_gemv ? mlp_hdrs.state_signature : 0)
                && cache.batch_size == batch_size;
            if (!reuse) {
                const hipError_t clear_err = clear_split_graph_cache(cache, true);
                if (clear_err != hipSuccess) {
                    return clear_err;
                }
                if (cache.stream == nullptr) {
                    launch_err = hipStreamCreate(&cache.stream);
                    if (launch_err == hipSuccess) {
                        cache.device_ordinal = device_ordinal;
                    }
                }
                bool capturing = false;
                if (launch_err == hipSuccess) {
                    launch_err = hipStreamBeginCapture(
                        cache.stream, hipStreamCaptureModeGlobal);
                    capturing = (launch_err == hipSuccess);
                }
                hipError_t rec_err = hipSuccess;
                if (capturing) {
                    rec_err = record_layers(cache.stream);
                    launch_err = hipStreamEndCapture(cache.stream, &cache.graph);
                    if (rec_err != hipSuccess && launch_err == hipSuccess) {
                        launch_err = rec_err;
                    }
                }
                if (launch_err == hipSuccess) {
                    launch_err = hipGraphInstantiate(
                        &cache.exec, cache.graph, nullptr, nullptr, 0);
                }
                if (launch_err == hipSuccess) {
                    cache.num_layers = num_layers;
                    cache.device_ordinal = device_ordinal;
                    cache.grid_in = grid_in;
                    cache.grid_mid = grid_mid;
                    cache.grid_out = grid_out;
                    cache.grid_gate = grid_gate;
                    cache.grid_up = grid_up;
                    cache.grid_down = grid_down;
                    cache.layers = layers;
                    cache.hidden_io = hidden_io;
                    cache.workspace = workspace;
                    cache.counters = counters;
                    cache.barrier_counter = barrier_counter;
                    cache.barrier_flag = barrier_flag;
                    cache.int4 = int4_scales;
                    cache.cos_table = cos_table;
                    cache.sin_table = sin_table;
                    cache.fp8_scales = fp8_scales;
                    cache.kv_fp8_descs = kv_fp8_descs;
                    cache.batch_descs = batch_descs;
                    cache.state_signature =
                        use_gqh_gemv ? mlp_hdrs.state_signature : 0;
                    cache.batch_size = batch_size;
                    static bool dumped_graph = false;
                    if (!dumped_graph) {
                        dumped_graph = true;
                        std::fprintf(stderr, "[hip-occ] gqh-split HIP graph captured\n");
                    }
                } else {
                    static bool dumped_graph_fail = false;
                    if (!dumped_graph_fail) {
                        dumped_graph_fail = true;
                        std::fprintf(
                            stderr,
                            "[hip-occ] gqh-split HIP graph capture failed: %s\n",
                            hipGetErrorString(launch_err));
                    }
                    const hipError_t clear_err = clear_split_graph_cache(cache, true);
                    if (clear_err != hipSuccess) {
                        return clear_err;
                    }
                }
            }
            if (g_hip_gqh_prepare_only) {
                g_hip_gqh_prepare_only = false;
                (void)hipGetLastError();
                const int prepare_status = persistent_decode_prepare_only_status(
                    hipDeviceSynchronize(), device_ordinal);
                std::fprintf(
                    stderr,
                    "[decode] GQH tight convert prepared%s\n",
                    launch_err == hipSuccess ? " + HIP graph" : " (eager decode)");
                return prepare_status;
            }
            if (launch_err == hipSuccess && cache.exec != nullptr) {
                launch_err = hipGraphLaunch(cache.exec, cache.stream);
                if (launch_err == hipSuccess) {
                    launch_err = hipStreamSynchronize(cache.stream);
                }
            } else if (launch_err != hipSuccess) {
                // Eager fallback if capture failed.
                launch_err = record_layers(0);
            }
        } else {
            if (g_hip_gqh_prepare_only) {
                g_hip_gqh_prepare_only = false;
                const int prepare_status = persistent_decode_prepare_only_status(
                    hipDeviceSynchronize(), device_ordinal);
                std::fprintf(
                    stderr,
                    "[decode] GQH tight convert prepared (eager decode)\n");
                return prepare_status;
            }
            launch_err = record_layers(0);
        }
    } else if (coop_requested && coop_supported && max_blocks_per_mp > 0) {
        // Args for cooperative launch: void** where each entry points to
        // local storage holding one argument value. Locals must stay alive
        // through the launch — they're destroyed at function exit, and
        // we call hipDeviceSynchronize before returning.
        const Qwen35DecodeLayerDesc* layers_typed =
            static_cast<const Qwen35DecodeLayerDesc*>(layers);
        T* hidden_io_typed = static_cast<T*>(hidden_io);
        const T* cos_typed = static_cast<const T*>(cos_table);
        const T* sin_typed = static_cast<const T*>(sin_table);
        const Qwen35FP8ScaleDesc* fp8_typed =
            static_cast<const Qwen35FP8ScaleDesc*>(fp8_scales);
        const KVCacheFp8Desc* kv_fp8_typed =
            static_cast<const KVCacheFp8Desc*>(kv_fp8_descs);
        const BatchSeqDesc* batch_descs_typed =
            static_cast<const BatchSeqDesc*>(batch_descs);
        const Qwen35INT4ScaleDesc* int4_typed =
            static_cast<const Qwen35INT4ScaleDesc*>(int4_scales);
        void* args[] = {
            &num_layers, &hidden_dim, &intermediate_size, &seqlen_offset,
            &layers_typed, &hidden_io_typed, &workspace, &counters,
            &barrier_counter, &barrier_flag,
            &timing_slots,
            &cos_typed, &sin_typed, &rotary_dim,
            &proj_buf_floats, &attn_scratch_floats, &enable_attention_trace,
            &fp8_typed, &kv_fp8_typed, &batch_size,
            &batch_descs_typed, &int4_typed, &io_flags,
        };

        launch_err = hipLaunchCooperativeKernel(
            kernel_fp,
            dim3(static_cast<unsigned int>(num_blocks)),
            dim3(block_size),
            args,
            static_cast<uint32_t>(shared_bytes),
            0);
    } else {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(supersonic_qwen35_persistent_decode_kernel<T>),
            dim3(static_cast<unsigned int>(num_blocks)),
            dim3(block_size),
            shared_bytes,
            0,
            num_layers,
            hidden_dim,
            intermediate_size,
            seqlen_offset,
            static_cast<const Qwen35DecodeLayerDesc*>(layers),
            static_cast<T*>(hidden_io),
            workspace,
            counters,
            barrier_counter,
            barrier_flag,
            timing_slots,
            static_cast<const T*>(cos_table),
            static_cast<const T*>(sin_table),
            rotary_dim,
            proj_buf_floats,
            attn_scratch_floats,
            enable_attention_trace,
            static_cast<const Qwen35FP8ScaleDesc*>(fp8_scales),
            static_cast<const KVCacheFp8Desc*>(kv_fp8_descs),
            batch_size,
            static_cast<const BatchSeqDesc*>(batch_descs),
            static_cast<const Qwen35INT4ScaleDesc*>(int4_scales),
            io_flags);
        launch_err = hipGetLastError();
    }

    const hipError_t sync_err = hipDeviceSynchronize();
    return persistent_decode_post_enqueue_status(
        launch_err, sync_err, device_ordinal);
}

// Restore conv+rec after fused B>1 verify to the prefix of `commit_len`
// tokens (1-based). Returns 0 if live linear state matches that prefix
// (including commit_len==B, already live). Returns 1 if no snapshot is
// available and the caller must sequential-replay.
extern "C" int supersonic_qwen35_hip_mtp_restore_linear_prefix(
    int device_ordinal, const void* layers, int commit_len) {
    DecodeBridgeLockGuard guard;
    // The snapshot and every live-state pointer belong to this ordinal. Do
    // the device switch before validating or copying any of those pointers so
    // a caller that last ran on another device cannot make HIP interpret the
    // address in the wrong context.
    ScopedHipDevice scoped(device_ordinal);
    if (!scoped.ok()) {
        return static_cast<int>(scoped.status);
    }
    auto finish = [&](int status) {
        const hipError_t restore_err = scoped.restore();
        return status != 0 ? status : static_cast<int>(restore_err);
    };
    MtpPrefixSnap& s = mtp_prefix_snap();
    if (commit_len <= 0 || s.device_ordinal != device_ordinal ||
        s.owner_layers != layers) {
        return finish(1);
    }
    if (!s.ready || s.n_b < 2 || s.n_layers <= 0) {
        return finish(1);
    }
    if (commit_len == s.n_b) {
        return finish(0);
    }
    if (commit_len > s.n_b) {
        return finish(1);
    }
    const int b = commit_len - 1;
    for (int layer = 0; layer < s.n_layers; ++layer) {
        if (s.rec_live[layer] != nullptr && s.rec[b][layer] != nullptr &&
            s.rec_bytes[layer] > 0) {
            hipError_t err = hipMemcpy(
                s.rec_live[layer],
                s.rec[b][layer],
                s.rec_bytes[layer],
                hipMemcpyDeviceToDevice);
            if (err != hipSuccess) {
                return finish(static_cast<int>(err));
            }
        }
        if (s.conv_live[layer] != nullptr && s.conv[b][layer] != nullptr &&
            s.conv_bytes[layer] > 0) {
            hipError_t err = hipMemcpy(
                s.conv_live[layer],
                s.conv[b][layer],
                s.conv_bytes[layer],
                hipMemcpyDeviceToDevice);
            if (err != hipSuccess) {
                return finish(static_cast<int>(err));
            }
        }
    }
    return finish(0);
}

static hipError_t clear_mtp_prefix_snap_if_owned(int device_ordinal, const void* layers) {
    MtpPrefixSnap& s = mtp_prefix_snap();
    if (layers == nullptr || s.owner_layers != layers ||
        s.device_ordinal != device_ordinal) {
        return hipSuccess;
    }
    const hipError_t err = release_mtp_prefix_slabs(
        s, s.rec_slab != nullptr, s.conv_slab != nullptr);
    if (err != hipSuccess) {
        return err;
    }
    s = MtpPrefixSnap{};
    return hipSuccess;
}

static int restore_gqh_projection(
    int device_ordinal,
    const GqhProjHdr& projection,
    int in_dim,
    int out_dim) {
    if (projection.wire == nullptr || projection.rung < 0) {
        return 0;
    }
    return supersonic_gqh_hip_restore_planar(
        device_ordinal,
        projection.rung,
        const_cast<void*>(projection.wire),
        in_dim,
        out_dim);
}

static int restore_gqh_layer_weights(
    const GqhMlpHdrs& cache,
    int hidden_dim,
    int intermediate_size) {
    for (int layer = 0; layer < cache.n; ++layer) {
        int status = restore_gqh_projection(
            cache.device_ordinal, cache.gate[layer], hidden_dim, intermediate_size);
        if (status == 0) {
            status = restore_gqh_projection(
                cache.device_ordinal, cache.up[layer], hidden_dim, intermediate_size);
        }
        if (status == 0) {
            status = restore_gqh_projection(
                cache.device_ordinal, cache.down[layer], intermediate_size, hidden_dim);
        }
        const GqhMixerLayer& mixer = cache.mix[layer];
        if (status == 0 && mixer.layer_type == 1) {
            status = restore_gqh_projection(
                cache.device_ordinal, mixer.q, hidden_dim, mixer.q_out);
            if (status == 0) {
                status = restore_gqh_projection(
                    cache.device_ordinal, mixer.k, hidden_dim, mixer.k_out);
            }
            if (status == 0) {
                status = restore_gqh_projection(
                    cache.device_ordinal, mixer.v, hidden_dim, mixer.k_out);
            }
            if (status == 0) {
                status = restore_gqh_projection(
                    cache.device_ordinal, mixer.o, mixer.attn_size, hidden_dim);
            }
        } else if (status == 0) {
            status = restore_gqh_projection(
                cache.device_ordinal, mixer.qkv, hidden_dim, mixer.qkv_out);
            if (status == 0) {
                status = restore_gqh_projection(
                    cache.device_ordinal, mixer.z, hidden_dim, mixer.z_out);
            }
            if (status == 0) {
                status = restore_gqh_projection(
                    cache.device_ordinal, mixer.lin_out, mixer.val_dim, hidden_dim);
            }
        }
        if (status != 0) {
            return status;
        }
    }
    return 0;
}

extern "C" int supersonic_qwen35_4b_hip_reset_decode_cache(
    int device_ordinal,
    const void* layers,
    const void* int4,
    int hidden_dim,
    int intermediate_size) {
    DecodeBridgeLockGuard guard;
    ScopedHipDevice scoped(device_ordinal);
    if (!scoped.ok()) {
        return static_cast<int>(scoped.status);
    }
    const bool mlp_owned = g_gqh_mlp_hdrs.ok &&
        g_gqh_mlp_hdrs.device_ordinal == device_ordinal &&
        g_gqh_mlp_hdrs.layers == layers && g_gqh_mlp_hdrs.int4 == int4;
    if (mlp_owned) {
        // Decode uses multiple streams. Quiesce all readers once, enqueue every
        // inverse transform on the owning device's default stream, then publish
        // the restored layout only after a single completion barrier.
        const hipError_t before = hipDeviceSynchronize();
        if (before != hipSuccess) {
            return static_cast<int>(before);
        }
        const int status = restore_gqh_layer_weights(
            g_gqh_mlp_hdrs, hidden_dim, intermediate_size);
        if (status != 0) {
            supersonic_gpu_integrity_fail_stop(
                "GQH reset planar restore", status, device_ordinal);
        }
        const hipError_t after = hipDeviceSynchronize();
        if (after != hipSuccess) {
            supersonic_gpu_integrity_fail_stop(
                "GQH reset planar restore sync",
                static_cast<int>(after),
                device_ordinal);
        }
    }
    const int invalidate_status = supersonic_qwen35_4b_hip_invalidate_decode_cache(
        device_ordinal, layers, int4);
    if (mlp_owned && invalidate_status != 0) {
        supersonic_gpu_integrity_fail_stop(
            "GQH reset cache invalidation", invalidate_status, device_ordinal);
    }
    return invalidate_status;
}

extern "C" int supersonic_qwen35_4b_hip_invalidate_decode_cache(
    int device_ordinal,
    const void* layers,
    const void* int4) {
    DecodeBridgeLockGuard guard;
    if (layers == nullptr && int4 == nullptr) {
        return 0;
    }

    const bool mlp_owned = g_gqh_mlp_hdrs.ok &&
        g_gqh_mlp_hdrs.device_ordinal == device_ordinal &&
        ((layers != nullptr && g_gqh_mlp_hdrs.layers == layers) ||
         (int4 != nullptr && g_gqh_mlp_hdrs.int4 == int4));
    const void* owned_layers = mlp_owned ? g_gqh_mlp_hdrs.layers : layers;
    const hipError_t mtp_err = clear_mtp_prefix_snap_if_owned(device_ordinal, owned_layers);
    if (mtp_err != hipSuccess) {
        return static_cast<int>(mtp_err);
    }

    SplitGraphCache& cache = g_split_graph_cache;
    const bool graph_owned = cache.device_ordinal == device_ordinal &&
        ((layers != nullptr && cache.layers == layers) ||
         (int4 != nullptr && cache.int4 == int4));
    if (!graph_owned) {
        if (mlp_owned) {
            g_gqh_mlp_hdrs = GqhMlpHdrs{};
        }
        return 0;
    }
    const hipError_t graph_err = clear_split_graph_cache(cache, true);
    if (graph_err != hipSuccess) {
        return static_cast<int>(graph_err);
    }
    if (mlp_owned) {
        g_gqh_mlp_hdrs = GqhMlpHdrs{};
    }
    return 0;
}

extern "C" int supersonic_qwen35_4b_hip_persistent_decode(
    int dtype,
    size_t device_ordinal,
    size_t num_layers,
    size_t hidden_dim,
    size_t intermediate_size,
    size_t seqlen_offset,
    const void* layers,
    void* hidden_io,
    float* workspace,
    unsigned int* counters,
    unsigned int* barrier_counter,
    unsigned int* barrier_flag,
    unsigned long long* timing_slots,
    const void* cos_table,
    const void* sin_table,
    size_t rotary_dim,
    size_t proj_buf_floats,
    size_t attn_scratch_floats,
    int enable_attention_trace,
    const void* fp8_scales,
    const void* kv_fp8_descs,
    size_t batch_size,
    const void* batch_descs,
    const void* int4_scales) {
    try {
        switch (dtype) {
        case 2:
            return persistent_decode_device<hip_bfloat16>(
                static_cast<int>(device_ordinal),
                static_cast<int>(num_layers),
                static_cast<int>(hidden_dim),
                static_cast<int>(intermediate_size),
                static_cast<int>(seqlen_offset),
                layers, hidden_io, workspace, counters,
                barrier_counter, barrier_flag, timing_slots,
                cos_table, sin_table, static_cast<int>(rotary_dim),
                static_cast<int>(proj_buf_floats),
                static_cast<int>(attn_scratch_floats),
                enable_attention_trace,
                fp8_scales,
                kv_fp8_descs,
                static_cast<int>(batch_size),
                batch_descs,
                int4_scales);
        default:
            return 256;
        }
    } catch (...) {
        // The decode body may have enqueued work before a trace/debug vector
        // grows. No C++ exception may cross this ABI; fail-stop before Rust
        // can unwind model-owned buffers.
        supersonic_gpu_integrity_fail_stop(
            "persistent decode host allocation", -1, static_cast<int>(device_ordinal));
    }
}

// BF16→FP8 KV cache quantization bridge
extern "C" int supersonic_qwen35_4b_hip_quantize_kv_to_fp8(
    int dtype,
    size_t device_ordinal,
    const void* src,
    void* dst_fp8,
    float* dst_scale,
    int num_kv_heads,
    int seq_len,
    int head_dim,
    int max_T,
    int pos_offset) {
    // KV-FP8 is outside the narrowed Qwen3.8 product contract. Keep this
    // historical symbol as a linker-visible rejection so retained exploratory
    // callers fail explicitly without enqueueing work on model-owned buffers.
    (void)dtype;
    (void)device_ordinal;
    (void)src;
    (void)dst_fp8;
    (void)dst_scale;
    (void)num_kv_heads;
    (void)seq_len;
    (void)head_dim;
    (void)max_T;
    (void)pos_offset;
    return 256;
}

// supersonic_query_gpu_info is in the 0.8B bridge, not duplicated here
