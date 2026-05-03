#![cfg(supersonic_backend_cuda)]

use gpu_hal::{
    copy_h2d, set_backend, sync, vmm_is_supported, Backend, ScalarType, VirtualAllocationRole,
    VirtualArena, VirtualBacking, VirtualBuffer,
};

fn require_cuda_vmm() -> bool {
    set_backend(Backend::Cuda);
    if !vmm_is_supported(Backend::Cuda, 0) {
        eprintln!("skip: CUDA VMM unsupported on this device/runtime");
        return false;
    }
    true
}

fn pattern_bytes(n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| (i as u8).wrapping_mul(17).wrapping_add(3))
        .collect()
}

fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter()
        .zip(b.iter())
        .position(|(left, right)| left != right)
        .or_else(|| (a.len() != b.len()).then_some(a.len().min(b.len())))
}

fn assert_bytes_eq(label: &str, got: &[u8], expected: &[u8]) {
    let diff = first_diff(got, expected);
    let window = diff.unwrap_or(0);
    let start = window.saturating_sub(8);
    let end = (window + 24).min(got.len()).min(expected.len());
    assert!(
        diff.is_none(),
        "{label} mismatch first_diff={diff:?} got_prefix={:?} expected_prefix={:?} got_at_diff={:?} expected_at_diff={:?}",
        &got[..got.len().min(16)],
        &expected[..expected.len().min(16)],
        &got[start..end],
        &expected[start..end]
    );
}

#[test]
fn cuda_vmm_reserve_map_round_trip_stable_pointer() {
    if !require_cuda_vmm() {
        return;
    }

    let mut buf = VirtualBuffer::reserve(0, ScalarType::U8, &[1 << 20], VirtualBacking::Discard)
        .expect("reserve CUDA virtual buffer");
    let base = buf.as_ptr();
    assert_eq!(buf.mapped_bytes(), 0);

    buf.map_prefix_bytes(4096).expect("map first page");
    assert_eq!(buf.as_ptr(), base, "mapping changed base VA");
    let first = pattern_bytes(4096);
    copy_h2d(0, buf.as_mut_ptr(), first.as_ptr() as *const _, first.len()).expect("H2D first");
    sync(0).expect("sync first");

    buf.map_prefix_bytes(128 * 1024).expect("map larger prefix");
    assert_eq!(buf.as_ptr(), base, "growing mapping changed base VA");
    let second = pattern_bytes(128 * 1024);
    copy_h2d(
        0,
        buf.as_mut_ptr(),
        second.as_ptr() as *const _,
        second.len(),
    )
    .expect("H2D second");
    sync(0).expect("sync second");
    assert_bytes_eq(
        "stable pointer remap",
        &buf.to_host_prefix_bytes(second.len()).expect("D2H"),
        &second,
    );
}

#[test]
fn cuda_vmm_sparse_range_round_trip_stable_pointer() {
    if !require_cuda_vmm() {
        return;
    }

    let mut buf = VirtualBuffer::reserve(0, ScalarType::U8, &[1 << 20], VirtualBacking::Discard)
        .expect("reserve CUDA virtual buffer");
    let base = buf.as_ptr();
    let first_offset = 0;
    let second_offset = 512 * 1024;
    let len = 4096;

    buf.map_range_bytes(first_offset, len)
        .expect("map first island");
    buf.map_range_bytes(second_offset, len)
        .expect("map second island");
    assert_eq!(buf.as_ptr(), base, "sparse mapping changed base VA");

    let first = pattern_bytes(len);
    let second = pattern_bytes(len)
        .into_iter()
        .map(|byte| byte.wrapping_add(91))
        .collect::<Vec<_>>();
    copy_h2d(
        0,
        buf.offset_ptr(first_offset) as *mut _,
        first.as_ptr() as *const _,
        first.len(),
    )
    .expect("H2D first island");
    copy_h2d(
        0,
        buf.offset_ptr(second_offset) as *mut _,
        second.as_ptr() as *const _,
        second.len(),
    )
    .expect("H2D second island");
    sync(0).expect("sync sparse H2D");

    assert_bytes_eq(
        "first island",
        &buf.to_host_range_bytes(first_offset, len)
            .expect("D2H first island"),
        &first,
    );
    assert_bytes_eq(
        "second island",
        &buf.to_host_range_bytes(second_offset, len)
            .expect("D2H second island"),
        &second,
    );
}

#[test]
fn cuda_vmm_cpu_backup_restore_round_trip() {
    if !require_cuda_vmm() {
        return;
    }

    let data = pattern_bytes(64 * 1024);
    let mut buf =
        VirtualBuffer::reserve(0, ScalarType::U8, &[data.len()], VirtualBacking::CpuBackup)
            .expect("reserve CUDA virtual buffer");
    buf.map_prefix_bytes(data.len()).expect("map prefix");
    copy_h2d(0, buf.as_mut_ptr(), data.as_ptr() as *const _, data.len()).expect("H2D");
    sync(0).expect("sync H2D");

    buf.backup_mapped_to_host().expect("backup");
    buf.unmap_all().expect("unmap");
    assert_eq!(buf.mapped_bytes(), 0);
    buf.restore_backup().expect("restore");
    sync(0).expect("sync restore");
    assert_bytes_eq("restored buffer", &buf.to_host_bytes().expect("D2H"), &data);
}

#[test]
fn cuda_vmm_cpu_evict_restore_packed_kv_round_trip() {
    if !require_cuda_vmm() {
        return;
    }

    let nkv = 2;
    let cap = 384;
    let head_dim = 256;
    let elem_bytes = ScalarType::BF16.size_in_bytes();
    let logical_len = 2 * nkv * cap * head_dim * elem_bytes;
    let half_len = logical_len / 2;
    let head_stride = cap * head_dim * elem_bytes;
    let prefix = 16 * head_dim * elem_bytes;

    let mut buf = VirtualBuffer::reserve(
        0,
        ScalarType::BF16,
        &[2, nkv, cap, head_dim],
        VirtualBacking::CpuBackup,
    )
    .expect("reserve packed KV CUDA virtual buffer");
    buf.map_prefix_bytes(logical_len).expect("map packed KV");

    let data = pattern_bytes(logical_len);
    copy_h2d(0, buf.as_mut_ptr(), data.as_ptr() as *const _, data.len()).expect("H2D packed KV");
    sync(0).expect("sync packed KV H2D");

    let before_k0 = buf
        .to_host_range_bytes(0, prefix)
        .expect("D2H K head 0 before");
    let before_k1 = buf
        .to_host_range_bytes(head_stride, prefix)
        .expect("D2H K head 1 before");
    let before_v0 = buf
        .to_host_range_bytes(half_len, prefix)
        .expect("D2H V head 0 before");
    let before_v1 = buf
        .to_host_range_bytes(half_len + head_stride, prefix)
        .expect("D2H V head 1 before");

    buf.evict_to_host().expect("evict packed KV");
    assert_eq!(buf.resident_bytes(), 0);
    assert_eq!(buf.logical_resident_bytes(), 0);
    assert_eq!(buf.mapping_count(), 0);
    assert_eq!(buf.logical_backup_bytes(), logical_len);
    buf.restore_backup().expect("restore packed KV");
    let stats = buf.stats();
    assert_eq!(stats.logical_bytes, logical_len);
    assert_eq!(stats.logical_resident_bytes, logical_len);
    assert_eq!(stats.logical_backup_bytes, logical_len);
    assert!(
        stats.resident_bytes >= stats.logical_resident_bytes,
        "physical residency must cover logical residency"
    );

    let restored = buf
        .to_host_prefix_bytes(logical_len)
        .expect("D2H restored packed KV");
    let checks = [
        ("K head 0", 0, before_k0),
        ("K head 1", head_stride, before_k1),
        ("V head 0", half_len, before_v0),
        ("V head 1", half_len + head_stride, before_v1),
    ];
    for (label, offset, expected) in checks {
        assert_bytes_eq(label, &restored[offset..offset + prefix], &expected);
    }
}

#[test]
fn cuda_vmm_arena_tracks_allocation_residency() {
    if !require_cuda_vmm() {
        return;
    }

    let mut arena = VirtualArena::new(0, VirtualBacking::CpuBackup);
    let k_id = arena
        .reserve(
            "layer0.k",
            VirtualAllocationRole::KvCache,
            ScalarType::BF16,
            &[1, 2, 384, 256],
        )
        .expect("reserve K allocation");
    let v_id = arena
        .reserve(
            "layer0.v",
            VirtualAllocationRole::KvCache,
            ScalarType::BF16,
            &[1, 2, 384, 256],
        )
        .expect("reserve V allocation");

    assert_eq!(arena.stats().allocations, 2);
    assert_eq!(
        arena.allocation(k_id).expect("K allocation").name(),
        "layer0.k"
    );
    assert_eq!(
        arena.allocation(v_id).expect("V allocation").role(),
        VirtualAllocationRole::KvCache
    );

    let logical_len = 2 * 384 * 256 * ScalarType::BF16.size_in_bytes();
    for id in [k_id, v_id] {
        let allocation = arena.allocation_mut(id).expect("allocation");
        allocation
            .buffer_mut()
            .map_prefix_bytes(logical_len)
            .expect("map allocation");
    }

    let stats = arena.stats();
    assert_eq!(stats.allocations, 2);
    assert_eq!(stats.logical_bytes, logical_len * 2);
    assert_eq!(stats.logical_resident_bytes, logical_len * 2);
    assert!(
        stats.resident_bytes >= stats.logical_resident_bytes,
        "physical residency must cover logical residency"
    );
}
