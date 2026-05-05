//! Parity test for `apply_rope_prefill_indirect` (SpecPrefill — Phase B).
//!
//! Two assertions:
//!
//!  1. **Identity parity**: when `pos_ids = [0, 1, ..., seq_len-1]`, the
//!     indirect kernel output is byte-identical to the dense
//!     `apply_rope_prefill` kernel. This is the load-bearing invariant —
//!     callers that don't yet know about SpecPrefill must be free to
//!     switch to the indirect kernel without producing different
//!     numerics.
//!
//!  2. **Sparse parity**: when `pos_ids` is a non-contiguous subset like
//!     `[3, 7, 12, 19]`, slot `i` of the output equals what the dense
//!     kernel produces for slot `pos_ids[i]` of a hypothetical full
//!     prompt — i.e. each kept token rotates by its source position's
//!     cos/sin row, not by its compacted slot index. Verified by running
//!     the dense kernel on a full prompt of length max(pos_ids)+1 and
//!     gathering the matching rows.
//!
//! Skipped silently when neither HIP nor CUDA is compiled.

use gpu_hal::{Backend, GpuBuffer, ScalarType};
use kernel_ffi::prefill_ffi::{apply_rope_prefill, apply_rope_prefill_indirect};

fn select_backend() -> Option<Backend> {
    if gpu_hal::is_backend_compiled(Backend::Hip) {
        Some(Backend::Hip)
    } else if gpu_hal::is_backend_compiled(Backend::Cuda) {
        Some(Backend::Cuda)
    } else {
        None
    }
}

/// Build a deterministic [seq_len, num_heads, head_dim] BF16 input on the GPU.
/// Values are bf16-rounded f32 sequence so the per-element rotation is
/// well-defined and reproducible.
fn make_data(
    ordinal: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    seed_offset: usize,
) -> GpuBuffer {
    let total = seq_len * num_heads * head_dim;
    let host: Vec<half::bf16> = (0..total)
        .map(|i| {
            let v = ((i + seed_offset) as f32 * 0.0017).sin() + 0.5;
            half::bf16::from_f32(v)
        })
        .collect();
    let mut buf = GpuBuffer::zeros(ordinal, ScalarType::BF16, &[seq_len, num_heads, head_dim])
        .expect("alloc data");
    let bytes = unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 2) };
    gpu_hal::copy_h2d(
        ordinal,
        buf.as_mut_ptr(),
        bytes.as_ptr() as *const _,
        bytes.len(),
    )
    .expect("h2d data");
    buf
}

/// Build cos/sin tables of shape [max_pos, half_rot] BF16. Frequencies are
/// the standard RoPE: theta_i = base^(-2i/rotary_dim). We bake the table
/// here rather than using a fixed model checkpoint so the test is
/// self-contained.
fn make_rope_tables(
    ordinal: usize,
    max_pos: usize,
    half_rot: usize,
    rotary_dim: usize,
    base: f32,
) -> (GpuBuffer, GpuBuffer) {
    let mut cos = vec![half::bf16::from_f32(0.0); max_pos * half_rot];
    let mut sin = vec![half::bf16::from_f32(0.0); max_pos * half_rot];
    for p in 0..max_pos {
        for i in 0..half_rot {
            let exponent = -2.0 * (i as f32) / (rotary_dim as f32);
            let freq = base.powf(exponent);
            let angle = (p as f32) * freq;
            cos[p * half_rot + i] = half::bf16::from_f32(angle.cos());
            sin[p * half_rot + i] = half::bf16::from_f32(angle.sin());
        }
    }
    let mut cos_buf =
        GpuBuffer::zeros(ordinal, ScalarType::BF16, &[max_pos, half_rot]).expect("alloc cos");
    let mut sin_buf =
        GpuBuffer::zeros(ordinal, ScalarType::BF16, &[max_pos, half_rot]).expect("alloc sin");
    let cos_bytes = unsafe { std::slice::from_raw_parts(cos.as_ptr() as *const u8, cos.len() * 2) };
    let sin_bytes = unsafe { std::slice::from_raw_parts(sin.as_ptr() as *const u8, sin.len() * 2) };
    gpu_hal::copy_h2d(
        ordinal,
        cos_buf.as_mut_ptr(),
        cos_bytes.as_ptr() as *const _,
        cos_bytes.len(),
    )
    .expect("h2d cos");
    gpu_hal::copy_h2d(
        ordinal,
        sin_buf.as_mut_ptr(),
        sin_bytes.as_ptr() as *const _,
        sin_bytes.len(),
    )
    .expect("h2d sin");
    (cos_buf, sin_buf)
}

fn upload_pos_ids(ordinal: usize, pos_ids: &[u32]) -> GpuBuffer {
    let mut buf =
        GpuBuffer::zeros(ordinal, ScalarType::U32, &[pos_ids.len()]).expect("alloc pos_ids");
    let bytes =
        unsafe { std::slice::from_raw_parts(pos_ids.as_ptr() as *const u8, pos_ids.len() * 4) };
    gpu_hal::copy_h2d(
        ordinal,
        buf.as_mut_ptr(),
        bytes.as_ptr() as *const _,
        bytes.len(),
    )
    .expect("h2d pos_ids");
    buf
}

fn d2h_bf16(buf: &GpuBuffer) -> Vec<half::bf16> {
    let elem_count = buf.elem_count();
    let mut bytes = vec![0u8; elem_count * 2];
    gpu_hal::copy_d2h(
        buf.device_ordinal(),
        bytes.as_mut_ptr() as *mut _,
        buf.as_ptr(),
        bytes.len(),
    )
    .expect("d2h");
    let mut out = vec![half::bf16::from_f32(0.0); elem_count];
    for (i, chunk) in bytes.chunks_exact(2).enumerate() {
        out[i] = half::bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]]));
    }
    out
}

#[test]
fn rope_indirect_identity_matches_dense() {
    let Some(backend) = select_backend() else {
        eprintln!("skipped: neither HIP nor CUDA backend compiled");
        return;
    };
    gpu_hal::set_backend(backend);
    let ordinal = 0usize;
    let seq_len = 16usize;
    let num_heads = 4usize;
    let head_dim = 64usize;
    let rotary_dim = 32usize;
    let half_rot = rotary_dim / 2;
    let max_pos = seq_len; // identity ⇒ pos_ids in [0, seq_len)
    let base = 10_000.0_f32;

    let (cos_buf, sin_buf) = make_rope_tables(ordinal, max_pos, half_rot, rotary_dim, base);

    // Two identical-input data buffers — one for dense, one for indirect.
    let mut data_dense = make_data(ordinal, seq_len, num_heads, head_dim, 0);
    let mut data_indirect = make_data(ordinal, seq_len, num_heads, head_dim, 0);

    apply_rope_prefill(
        ordinal,
        ScalarType::BF16,
        seq_len,
        num_heads,
        head_dim,
        rotary_dim,
        &cos_buf,
        &sin_buf,
        0,
        &mut data_dense,
    )
    .expect("dense rope");

    let pos_ids: Vec<u32> = (0..seq_len as u32).collect();
    let pos_ids_buf = upload_pos_ids(ordinal, &pos_ids);
    apply_rope_prefill_indirect(
        ordinal,
        ScalarType::BF16,
        seq_len,
        num_heads,
        head_dim,
        rotary_dim,
        &cos_buf,
        &sin_buf,
        &pos_ids_buf,
        &mut data_indirect,
    )
    .expect("indirect rope");

    let d = d2h_bf16(&data_dense);
    let i = d2h_bf16(&data_indirect);
    assert_eq!(d.len(), i.len());
    for k in 0..d.len() {
        assert_eq!(
            d[k].to_bits(),
            i[k].to_bits(),
            "identity-pos_ids parity broke at element {k}: dense={} indirect={}",
            d[k].to_f32(),
            i[k].to_f32()
        );
    }
}

#[test]
fn rope_indirect_sparse_matches_gathered_dense() {
    let Some(backend) = select_backend() else {
        eprintln!("skipped: neither HIP nor CUDA backend compiled");
        return;
    };
    gpu_hal::set_backend(backend);
    let ordinal = 0usize;
    let num_heads = 4usize;
    let head_dim = 64usize;
    let rotary_dim = 32usize;
    let half_rot = rotary_dim / 2;
    // Non-contiguous prompt positions; max(pos_ids) = 19 ⇒ table needs 20 rows.
    let pos_ids: Vec<u32> = vec![3, 7, 12, 19];
    let seq_kept = pos_ids.len();
    let max_pos = (*pos_ids.iter().max().unwrap() as usize) + 1;
    let base = 10_000.0_f32;

    let (cos_buf, sin_buf) = make_rope_tables(ordinal, max_pos, half_rot, rotary_dim, base);

    // Build a "full" data tensor of length max_pos with the same per-element
    // value rule, so slot `pos_ids[i]` of the dense output corresponds to
    // slot `i` of the kept-tokens input.
    //
    // The kept-token data must be the exact subset: kept_data[i] = full[pos_ids[i]].
    let full_total = max_pos * num_heads * head_dim;
    let full_host: Vec<half::bf16> = (0..full_total)
        .map(|j| half::bf16::from_f32(((j as f32) * 0.0017).sin() + 0.5))
        .collect();

    let mut full_data =
        GpuBuffer::zeros(ordinal, ScalarType::BF16, &[max_pos, num_heads, head_dim])
            .expect("alloc full");
    let full_bytes =
        unsafe { std::slice::from_raw_parts(full_host.as_ptr() as *const u8, full_host.len() * 2) };
    gpu_hal::copy_h2d(
        ordinal,
        full_data.as_mut_ptr(),
        full_bytes.as_ptr() as *const _,
        full_bytes.len(),
    )
    .expect("h2d full");

    // Kept input: gather rows pos_ids[i] from full_host.
    let row_elems = num_heads * head_dim;
    let mut kept_host = vec![half::bf16::from_f32(0.0); seq_kept * row_elems];
    for (i, &p) in pos_ids.iter().enumerate() {
        let p = p as usize;
        kept_host[i * row_elems..(i + 1) * row_elems]
            .copy_from_slice(&full_host[p * row_elems..(p + 1) * row_elems]);
    }
    let mut kept_data =
        GpuBuffer::zeros(ordinal, ScalarType::BF16, &[seq_kept, num_heads, head_dim])
            .expect("alloc kept");
    let kept_bytes =
        unsafe { std::slice::from_raw_parts(kept_host.as_ptr() as *const u8, kept_host.len() * 2) };
    gpu_hal::copy_h2d(
        ordinal,
        kept_data.as_mut_ptr(),
        kept_bytes.as_ptr() as *const _,
        kept_bytes.len(),
    )
    .expect("h2d kept");

    // Reference: dense rope on the FULL tensor.
    apply_rope_prefill(
        ordinal,
        ScalarType::BF16,
        max_pos,
        num_heads,
        head_dim,
        rotary_dim,
        &cos_buf,
        &sin_buf,
        0,
        &mut full_data,
    )
    .expect("dense rope full");

    // Indirect: rope on the kept tensor with pos_ids.
    let pos_ids_buf = upload_pos_ids(ordinal, &pos_ids);
    apply_rope_prefill_indirect(
        ordinal,
        ScalarType::BF16,
        seq_kept,
        num_heads,
        head_dim,
        rotary_dim,
        &cos_buf,
        &sin_buf,
        &pos_ids_buf,
        &mut kept_data,
    )
    .expect("indirect rope kept");

    let full = d2h_bf16(&full_data);
    let kept = d2h_bf16(&kept_data);

    for (i, &p) in pos_ids.iter().enumerate() {
        let p = p as usize;
        for k in 0..row_elems {
            let want = full[p * row_elems + k];
            let got = kept[i * row_elems + k];
            assert_eq!(
                want.to_bits(),
                got.to_bits(),
                "sparse parity broke at kept-slot {i} (orig pos {p}), element {k}: \
                 want={} got={}",
                want.to_f32(),
                got.to_f32()
            );
        }
    }
}
