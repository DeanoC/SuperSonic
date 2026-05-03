#![cfg(supersonic_backend_hip)]

use gpu_hal::{
    copy_h2d, set_backend, sync, vmm_is_supported, Backend, ScalarType, VirtualAllocationRole,
    VirtualArena, VirtualBacking, VirtualBuffer,
};

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
fn vmm_reserve_map_round_trip_stable_pointer() {
    set_backend(Backend::Hip);
    if !vmm_is_supported(Backend::Hip, 0) {
        eprintln!("skip: HIP VMM unsupported on this device/runtime");
        return;
    }

    let mut buf = VirtualBuffer::reserve(0, ScalarType::U8, &[1 << 20], VirtualBacking::Discard)
        .expect("reserve virtual buffer");
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
    assert_eq!(buf.to_host_prefix_bytes(second.len()).expect("D2H"), second);
}

#[test]
fn vmm_sparse_range_round_trip_stable_pointer() {
    set_backend(Backend::Hip);
    if !vmm_is_supported(Backend::Hip, 0) {
        eprintln!("skip: HIP VMM unsupported on this device/runtime");
        return;
    }

    let mut buf = VirtualBuffer::reserve(0, ScalarType::U8, &[1 << 20], VirtualBacking::Discard)
        .expect("reserve virtual buffer");
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

    assert_eq!(
        buf.to_host_range_bytes(first_offset, len)
            .expect("D2H first island"),
        first
    );
    assert_eq!(
        buf.to_host_range_bytes(second_offset, len)
            .expect("D2H second island"),
        second
    );
    if buf.granularity() <= second_offset {
        assert!(
            buf.to_host_prefix_bytes(second_offset + len).is_err(),
            "sparse mapping should not be readable as a contiguous prefix"
        );
    } else {
        assert!(
            buf.to_host_prefix_bytes(second_offset + len).is_ok(),
            "coarse HIP VMM pages map both test islands into one readable prefix"
        );
    }
}

#[test]
fn vmm_large_single_map_round_trip() {
    set_backend(Backend::Hip);
    if !vmm_is_supported(Backend::Hip, 0) {
        eprintln!("skip: HIP VMM unsupported on this device/runtime");
        return;
    }

    let len = 512 * 1024;
    let mut buf = VirtualBuffer::reserve(0, ScalarType::U8, &[len], VirtualBacking::Discard)
        .expect("reserve virtual buffer");
    buf.map_prefix_bytes(len).expect("map large prefix");
    let data = pattern_bytes(len);
    copy_h2d(0, buf.as_mut_ptr(), data.as_ptr() as *const _, data.len()).expect("H2D");
    sync(0).expect("sync large H2D");
    assert_eq!(buf.to_host_prefix_bytes(len).expect("D2H"), data);
}

#[test]
fn vmm_two_buffers_map_round_trip() {
    set_backend(Backend::Hip);
    if !vmm_is_supported(Backend::Hip, 0) {
        eprintln!("skip: HIP VMM unsupported on this device/runtime");
        return;
    }

    let len = 64 * 1024;
    let mut a = VirtualBuffer::reserve(
        0,
        ScalarType::BF16,
        &[1, 2, 512, 256],
        VirtualBacking::CpuBackup,
    )
    .expect("reserve first virtual buffer");
    let mut b = VirtualBuffer::reserve(
        0,
        ScalarType::BF16,
        &[1, 2, 512, 256],
        VirtualBacking::CpuBackup,
    )
    .expect("reserve second virtual buffer");
    a.map_range_bytes(0, len).expect("map first buffer");
    b.map_range_bytes(0, len).expect("map second buffer");
}

#[test]
fn vmm_many_kv_buffers_map_round_trip() {
    set_backend(Backend::Hip);
    if !vmm_is_supported(Backend::Hip, 0) {
        eprintln!("skip: HIP VMM unsupported on this device/runtime");
        return;
    }

    let len = 8 * 256 * ScalarType::BF16.size_in_bytes();
    let head_stride = 512 * 256 * ScalarType::BF16.size_in_bytes();
    let value_base = 2 * head_stride;
    let mut buffers = Vec::new();
    for _ in 0..16 {
        buffers.push(
            VirtualBuffer::reserve(
                0,
                ScalarType::BF16,
                &[2, 2, 512, 256],
                VirtualBacking::CpuBackup,
            )
            .expect("reserve KV-like virtual buffer"),
        );
    }
    for (idx, buf) in buffers.iter_mut().enumerate() {
        buf.map_range_bytes(0, len)
            .unwrap_or_else(|e| panic!("map KV-like virtual buffer {idx}: {e}"));
        buf.map_range_bytes(head_stride, len)
            .unwrap_or_else(|e| panic!("map KV-like virtual buffer {idx} second head: {e}"));
        buf.map_range_bytes(value_base, len)
            .unwrap_or_else(|e| panic!("map KV-like virtual buffer {idx} value: {e}"));
        buf.map_range_bytes(value_base + head_stride, len)
            .unwrap_or_else(|e| panic!("map KV-like virtual buffer {idx} second value: {e}"));
    }
}

#[test]
fn vmm_cpu_backup_restore_round_trip() {
    set_backend(Backend::Hip);
    if !vmm_is_supported(Backend::Hip, 0) {
        eprintln!("skip: HIP VMM unsupported on this device/runtime");
        return;
    }

    let data = pattern_bytes(64 * 1024);
    let mut buf =
        VirtualBuffer::reserve(0, ScalarType::U8, &[data.len()], VirtualBacking::CpuBackup)
            .expect("reserve virtual buffer");
    buf.map_prefix_bytes(data.len()).expect("map prefix");
    copy_h2d(0, buf.as_mut_ptr(), data.as_ptr() as *const _, data.len()).expect("H2D");
    sync(0).expect("sync H2D");

    buf.backup_mapped_to_host().expect("backup");
    buf.unmap_all().expect("unmap");
    assert_eq!(buf.mapped_bytes(), 0);
    buf.restore_backup().expect("restore");
    sync(0).expect("sync restore");
    assert_eq!(buf.to_host_bytes().expect("D2H"), data);
}

#[test]
fn vmm_cpu_evict_restore_packed_kv_round_trip() {
    set_backend(Backend::Hip);
    if !vmm_is_supported(Backend::Hip, 0) {
        eprintln!("skip: HIP VMM unsupported on this device/runtime");
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
    .expect("reserve packed KV virtual buffer");
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
        let got = &restored[offset..offset + prefix];
        assert!(
            first_diff(got, &expected).is_none(),
            "{label} mismatch first_diff={:?}",
            first_diff(got, &expected)
        );
    }
}

#[test]
fn vmm_arena_tracks_allocation_residency() {
    set_backend(Backend::Hip);
    if !vmm_is_supported(Backend::Hip, 0) {
        eprintln!("skip: HIP VMM unsupported on this device/runtime");
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
    assert_eq!(stats.logical_bytes, 2 * logical_len);
    assert_eq!(stats.logical_resident_bytes, 2 * logical_len);
    assert_eq!(stats.mapping_count, 2);
    assert!(stats.resident_bytes >= stats.logical_resident_bytes);
    assert!(stats.reserved_bytes >= stats.resident_bytes);

    let per_alloc = arena.allocation_stats();
    assert_eq!(per_alloc.len(), 2);
    assert_eq!(per_alloc[0].role, VirtualAllocationRole::KvCache);
    assert_eq!(per_alloc[0].buffer.logical_resident_bytes, logical_len);
}

#[test]
fn vmm_arena_tags_weight_and_moe_allocations() {
    set_backend(Backend::Hip);
    if !vmm_is_supported(Backend::Hip, 0) {
        eprintln!("skip: HIP VMM unsupported on this device/runtime");
        return;
    }

    let mut arena = VirtualArena::new(0, VirtualBacking::Discard);
    let weight_id = arena
        .reserve(
            "layer0.w_pack",
            VirtualAllocationRole::Weights,
            ScalarType::U8,
            &[4096],
        )
        .expect("reserve virtual weight allocation");
    let expert_id = arena
        .reserve(
            "moe.expert17",
            VirtualAllocationRole::MoeExpert,
            ScalarType::U8,
            &[4096],
        )
        .expect("reserve virtual MoE expert allocation");

    assert_eq!(
        arena.allocation(weight_id).expect("weight").role(),
        VirtualAllocationRole::Weights
    );
    assert_eq!(
        arena.allocation(expert_id).expect("expert").role(),
        VirtualAllocationRole::MoeExpert
    );
    assert_eq!(arena.stats().allocations, 2);
    assert_eq!(arena.stats().logical_bytes, 8192);
    assert_eq!(arena.stats().resident_bytes, 0);
}

#[test]
#[ignore = "HIP VMM repeated eviction is order-sensitive when mixed with other VMM tests"]
fn vmm_split_kv_repeated_backup_restore_stress() {
    set_backend(Backend::Hip);
    if !vmm_is_supported(Backend::Hip, 0) {
        eprintln!("skip: HIP VMM unsupported on this device/runtime");
        return;
    }
    run_split_kv_evict_restore_stress(1, 2);
}

#[derive(Debug, Clone, Copy)]
enum SplitKvRestorePattern {
    Pairwise,
    EvictAllRestorePairs,
    EvictAllRestoreAllKThenAllV,
    EvictAllRestoreReversePairs,
    EvictAllRestorePairsThenMapPressure,
}

struct SplitKvBufferPair {
    k_buf: VirtualBuffer,
    v_buf: VirtualBuffer,
    k_data: Vec<u8>,
    v_data: Vec<u8>,
}

#[test]
#[ignore = "HIP VMM characterization: run explicitly to categorize split-KV remap/restore ordering"]
fn vmm_split_kv_remap_restore_order_matrix() {
    set_backend(Backend::Hip);
    if !vmm_is_supported(Backend::Hip, 0) {
        eprintln!("skip: HIP VMM unsupported on this device/runtime");
        return;
    }

    let cases = [
        (SplitKvRestorePattern::Pairwise, 1, 3),
        (SplitKvRestorePattern::Pairwise, 6, 2),
        (SplitKvRestorePattern::EvictAllRestorePairs, 6, 2),
        (SplitKvRestorePattern::EvictAllRestoreAllKThenAllV, 6, 2),
        (SplitKvRestorePattern::EvictAllRestoreReversePairs, 6, 2),
        (
            SplitKvRestorePattern::EvictAllRestorePairsThenMapPressure,
            6,
            2,
        ),
    ];

    for (pattern, buffer_count, rounds) in cases {
        eprintln!("case pattern={pattern:?} buffers={buffer_count} rounds={rounds}");
        run_split_kv_restore_pattern(buffer_count, rounds, pattern);
    }
}

#[test]
#[ignore = "HIP VMM characterization: production-like all-evict/all-restore split-KV stress"]
fn vmm_split_kv_all_evict_restore_pairs_stress() {
    set_backend(Backend::Hip);
    if !vmm_is_supported(Backend::Hip, 0) {
        eprintln!("skip: HIP VMM unsupported on this device/runtime");
        return;
    }
    run_split_kv_restore_pattern(6, 4, SplitKvRestorePattern::EvictAllRestorePairs);
}

#[test]
#[ignore = "HIP VMM characterization: backup all split-KV buffers before unmapping any of them"]
fn vmm_split_kv_two_phase_backup_unmap_restore_stress() {
    set_backend(Backend::Hip);
    if !vmm_is_supported(Backend::Hip, 0) {
        eprintln!("skip: HIP VMM unsupported on this device/runtime");
        return;
    }
    run_split_kv_two_phase_backup_unmap_restore(6, 4, false);
}

#[test]
#[ignore = "HIP VMM characterization: backup all split-KV buffers and D2H-fence before unmapping"]
fn vmm_split_kv_two_phase_backup_fence_unmap_restore_stress() {
    set_backend(Backend::Hip);
    if !vmm_is_supported(Backend::Hip, 0) {
        eprintln!("skip: HIP VMM unsupported on this device/runtime");
        return;
    }
    run_split_kv_two_phase_backup_unmap_restore(6, 4, true);
}

#[test]
#[ignore = "HIP VMM characterization: map all restore ranges before copying any backups"]
fn vmm_split_kv_two_phase_map_all_then_copy_restore_stress() {
    set_backend(Backend::Hip);
    if !vmm_is_supported(Backend::Hip, 0) {
        eprintln!("skip: HIP VMM unsupported on this device/runtime");
        return;
    }
    run_split_kv_two_phase_map_all_then_copy_restore(6, 4);
}

#[test]
fn vmm_split_kv_two_phase_map_all_then_copy_restore_regression() {
    set_backend(Backend::Hip);
    if !vmm_is_supported(Backend::Hip, 0) {
        eprintln!("skip: HIP VMM unsupported on this device/runtime");
        return;
    }
    run_split_kv_two_phase_map_all_then_copy_restore(6, 1);
}

#[test]
#[ignore = "HIP VMM characterization: verify whether many live split-KV VMM buffers corrupt before eviction"]
fn vmm_split_kv_many_live_initial_fill_stress() {
    set_backend(Backend::Hip);
    if !vmm_is_supported(Backend::Hip, 0) {
        eprintln!("skip: HIP VMM unsupported on this device/runtime");
        return;
    }
    drop(make_split_kv_pairs(6));
}

#[test]
#[ignore = "HIP VMM characterization: pairwise eviction with six live split-KV pairs"]
fn vmm_split_kv_pairwise_six_live_pairs_stress() {
    set_backend(Backend::Hip);
    if !vmm_is_supported(Backend::Hip, 0) {
        eprintln!("skip: HIP VMM unsupported on this device/runtime");
        return;
    }
    run_split_kv_restore_pattern(6, 2, SplitKvRestorePattern::Pairwise);
}

#[test]
#[ignore = "HIP VMM characterization: pairwise eviction with two live split-KV pairs"]
fn vmm_split_kv_pairwise_two_live_pairs_stress() {
    set_backend(Backend::Hip);
    if !vmm_is_supported(Backend::Hip, 0) {
        eprintln!("skip: HIP VMM unsupported on this device/runtime");
        return;
    }
    run_split_kv_restore_pattern(2, 2, SplitKvRestorePattern::Pairwise);
}

#[test]
#[ignore = "HIP VMM characterization: distinguish backup capture corruption from restore corruption"]
fn vmm_split_kv_two_pair_backup_capture_diagnostic() {
    set_backend(Backend::Hip);
    if !vmm_is_supported(Backend::Hip, 0) {
        eprintln!("skip: HIP VMM unsupported on this device/runtime");
        return;
    }

    let (_nkv, _cap, _head_dim, _elem_bytes, logical_len) = split_kv_geometry();
    let mut buffers = make_split_kv_pairs(2);
    for idx in 0..buffers.len() {
        verify_split_kv_pairs(&format!("before backup pair {idx}"), &buffers);
        buffers[idx]
            .k_buf
            .backup_mapped_to_host()
            .unwrap_or_else(|e| panic!("backup K buffer {idx}: {e}"));
        assert_backup_eq(
            &format!("captured K backup {idx}"),
            &buffers[idx].k_buf,
            logical_len,
            &buffers[idx].k_data,
        );
        buffers[idx]
            .v_buf
            .backup_mapped_to_host()
            .unwrap_or_else(|e| panic!("backup V buffer {idx}: {e}"));
        assert_backup_eq(
            &format!("captured V backup {idx}"),
            &buffers[idx].v_buf,
            logical_len,
            &buffers[idx].v_data,
        );
        verify_split_kv_pairs(&format!("after backup pair {idx}"), &buffers);

        buffers[idx]
            .k_buf
            .evict_discard()
            .unwrap_or_else(|e| panic!("discard K buffer {idx}: {e}"));
        buffers[idx]
            .v_buf
            .evict_discard()
            .unwrap_or_else(|e| panic!("discard V buffer {idx}: {e}"));
        buffers[idx]
            .k_buf
            .restore_backup()
            .unwrap_or_else(|e| panic!("restore K buffer {idx}: {e}"));
        buffers[idx]
            .v_buf
            .restore_backup()
            .unwrap_or_else(|e| panic!("restore V buffer {idx}: {e}"));
        verify_split_kv_pairs(&format!("after restore pair {idx}"), &buffers);
    }
}

#[test]
#[ignore = "HIP VMM characterization: inspect evict_to_host backup before restore"]
fn vmm_split_kv_two_pair_evict_to_host_backup_diagnostic() {
    set_backend(Backend::Hip);
    if !vmm_is_supported(Backend::Hip, 0) {
        eprintln!("skip: HIP VMM unsupported on this device/runtime");
        return;
    }

    let (_nkv, _cap, _head_dim, _elem_bytes, logical_len) = split_kv_geometry();
    let mut buffers = make_split_kv_pairs(2);
    for idx in 0..buffers.len() {
        verify_split_kv_pairs(&format!("evict_to_host before pair {idx}"), &buffers);
        buffers[idx]
            .k_buf
            .evict_to_host()
            .unwrap_or_else(|e| panic!("evict_to_host K buffer {idx}: {e}"));
        assert_backup_eq(
            &format!("evict_to_host captured K backup {idx}"),
            &buffers[idx].k_buf,
            logical_len,
            &buffers[idx].k_data,
        );
        buffers[idx]
            .v_buf
            .evict_to_host()
            .unwrap_or_else(|e| panic!("evict_to_host V buffer {idx}: {e}"));
        assert_backup_eq(
            &format!("evict_to_host captured V backup {idx}"),
            &buffers[idx].v_buf,
            logical_len,
            &buffers[idx].v_data,
        );
        buffers[idx]
            .k_buf
            .restore_backup()
            .unwrap_or_else(|e| panic!("restore K buffer {idx}: {e}"));
        buffers[idx]
            .v_buf
            .restore_backup()
            .unwrap_or_else(|e| panic!("restore V buffer {idx}: {e}"));
        verify_split_kv_pairs(&format!("evict_to_host after restore pair {idx}"), &buffers);
    }
}

#[test]
#[ignore = "HIP VMM characterization: manual backup followed immediately by discard-unmap"]
fn vmm_split_kv_two_pair_manual_backup_immediate_unmap_diagnostic() {
    set_backend(Backend::Hip);
    if !vmm_is_supported(Backend::Hip, 0) {
        eprintln!("skip: HIP VMM unsupported on this device/runtime");
        return;
    }
    run_split_kv_manual_backup_unmap_diagnostic(false);
}

#[test]
#[ignore = "HIP VMM characterization: manual backup plus D2H read fence before discard-unmap"]
fn vmm_split_kv_two_pair_manual_backup_read_fence_unmap_diagnostic() {
    set_backend(Backend::Hip);
    if !vmm_is_supported(Backend::Hip, 0) {
        eprintln!("skip: HIP VMM unsupported on this device/runtime");
        return;
    }
    run_split_kv_manual_backup_unmap_diagnostic(true);
}

#[test]
#[ignore = "HIP VMM characterization: pairwise eviction with three live split-KV pairs"]
fn vmm_split_kv_pairwise_three_live_pairs_stress() {
    set_backend(Backend::Hip);
    if !vmm_is_supported(Backend::Hip, 0) {
        eprintln!("skip: HIP VMM unsupported on this device/runtime");
        return;
    }
    run_split_kv_restore_pattern(3, 2, SplitKvRestorePattern::Pairwise);
}

#[test]
#[ignore = "HIP VMM characterization: all K buffers restored before all V buffers"]
fn vmm_split_kv_restore_all_k_then_all_v_stress() {
    set_backend(Backend::Hip);
    if !vmm_is_supported(Backend::Hip, 0) {
        eprintln!("skip: HIP VMM unsupported on this device/runtime");
        return;
    }
    run_split_kv_restore_pattern(6, 2, SplitKvRestorePattern::EvictAllRestoreAllKThenAllV);
}

#[test]
#[ignore = "HIP VMM characterization: reverse-pair restore ordering"]
fn vmm_split_kv_restore_reverse_pairs_stress() {
    set_backend(Backend::Hip);
    if !vmm_is_supported(Backend::Hip, 0) {
        eprintln!("skip: HIP VMM unsupported on this device/runtime");
        return;
    }
    run_split_kv_restore_pattern(6, 2, SplitKvRestorePattern::EvictAllRestoreReversePairs);
}

#[test]
#[ignore = "HIP VMM characterization: test whether later mappings corrupt restored split-KV buffers"]
fn vmm_split_kv_post_restore_mapping_pressure_stress() {
    set_backend(Backend::Hip);
    if !vmm_is_supported(Backend::Hip, 0) {
        eprintln!("skip: HIP VMM unsupported on this device/runtime");
        return;
    }
    run_split_kv_restore_pattern(
        6,
        4,
        SplitKvRestorePattern::EvictAllRestorePairsThenMapPressure,
    );
}

fn run_split_kv_evict_restore_stress(buffer_count: usize, rounds: usize) {
    run_split_kv_restore_pattern(buffer_count, rounds, SplitKvRestorePattern::Pairwise);
}

fn split_kv_geometry() -> (usize, usize, usize, usize, usize) {
    let nkv = 2;
    let cap = 384;
    let head_dim = 256;
    let elem_bytes = ScalarType::BF16.size_in_bytes();
    let logical_len = nkv * cap * head_dim * elem_bytes;
    (nkv, cap, head_dim, elem_bytes, logical_len)
}

fn assert_backup_eq(label: &str, buf: &VirtualBuffer, len: usize, expected: &[u8]) {
    let chunks = buf
        .backup_chunks_for_debug()
        .unwrap_or_else(|| panic!("{label}: no backup chunks captured"));
    let mut got = vec![0u8; len];
    for (offset, data) in chunks {
        let end = offset + data.len();
        assert!(
            end <= got.len(),
            "{label}: backup chunk [{offset}, {end}) exceeds logical len {}",
            got.len()
        );
        got[offset..end].copy_from_slice(&data);
    }
    assert_bytes_eq(label, &got, expected);
}

fn run_split_kv_manual_backup_unmap_diagnostic(read_fence: bool) {
    let (_nkv, _cap, _head_dim, _elem_bytes, logical_len) = split_kv_geometry();
    let mut buffers = make_split_kv_pairs(2);
    for idx in 0..buffers.len() {
        verify_split_kv_pairs(&format!("manual before backup pair {idx}"), &buffers);
        buffers[idx]
            .k_buf
            .backup_mapped_to_host()
            .unwrap_or_else(|e| panic!("manual backup K buffer {idx}: {e}"));
        buffers[idx]
            .v_buf
            .backup_mapped_to_host()
            .unwrap_or_else(|e| panic!("manual backup V buffer {idx}: {e}"));
        if read_fence {
            let k = buffers[idx]
                .k_buf
                .to_host_range_bytes(0, 4096)
                .unwrap_or_else(|e| panic!("manual read fence K buffer {idx}: {e}"));
            assert_bytes_eq(
                &format!("manual read fence K buffer {idx}"),
                &k,
                &buffers[idx].k_data[..4096],
            );
            let v = buffers[idx]
                .v_buf
                .to_host_range_bytes(0, 4096)
                .unwrap_or_else(|e| panic!("manual read fence V buffer {idx}: {e}"));
            assert_bytes_eq(
                &format!("manual read fence V buffer {idx}"),
                &v,
                &buffers[idx].v_data[..4096],
            );
        }
        assert_backup_eq(
            &format!("manual captured K backup {idx}"),
            &buffers[idx].k_buf,
            logical_len,
            &buffers[idx].k_data,
        );
        assert_backup_eq(
            &format!("manual captured V backup {idx}"),
            &buffers[idx].v_buf,
            logical_len,
            &buffers[idx].v_data,
        );
        buffers[idx]
            .k_buf
            .evict_discard()
            .unwrap_or_else(|e| panic!("manual discard K buffer {idx}: {e}"));
        buffers[idx]
            .v_buf
            .evict_discard()
            .unwrap_or_else(|e| panic!("manual discard V buffer {idx}: {e}"));
        buffers[idx]
            .k_buf
            .restore_backup()
            .unwrap_or_else(|e| panic!("manual restore K buffer {idx}: {e}"));
        buffers[idx]
            .v_buf
            .restore_backup()
            .unwrap_or_else(|e| panic!("manual restore V buffer {idx}: {e}"));
        verify_split_kv_pairs(&format!("manual after restore pair {idx}"), &buffers);
    }
}

fn run_split_kv_two_phase_backup_unmap_restore(
    buffer_count: usize,
    rounds: usize,
    read_fence: bool,
) {
    let (_nkv, _cap, _head_dim, _elem_bytes, logical_len) = split_kv_geometry();
    let mut buffers = make_split_kv_pairs(buffer_count);
    for round in 0..rounds {
        eprintln!("two-phase round={round} buffers={buffer_count} read_fence={read_fence}");
        verify_split_kv_pairs(&format!("two-phase round {round} before backup"), &buffers);
        for (idx, pair) in buffers.iter_mut().enumerate() {
            pair.k_buf
                .backup_mapped_to_host()
                .unwrap_or_else(|e| panic!("two-phase backup K buffer {idx} round {round}: {e}"));
            pair.v_buf
                .backup_mapped_to_host()
                .unwrap_or_else(|e| panic!("two-phase backup V buffer {idx} round {round}: {e}"));
        }
        for (idx, pair) in buffers.iter().enumerate() {
            assert_backup_eq(
                &format!("two-phase captured K backup {idx} round {round}"),
                &pair.k_buf,
                logical_len,
                &pair.k_data,
            );
            assert_backup_eq(
                &format!("two-phase captured V backup {idx} round {round}"),
                &pair.v_buf,
                logical_len,
                &pair.v_data,
            );
        }
        if read_fence {
            for (idx, pair) in buffers.iter().enumerate() {
                let k = pair
                    .k_buf
                    .to_host_range_bytes(0, 4096)
                    .unwrap_or_else(|e| panic!("two-phase read fence K {idx} round {round}: {e}"));
                assert_bytes_eq(
                    &format!("two-phase read fence K {idx} round {round}"),
                    &k,
                    &pair.k_data[..4096],
                );
                let v = pair
                    .v_buf
                    .to_host_range_bytes(0, 4096)
                    .unwrap_or_else(|e| panic!("two-phase read fence V {idx} round {round}: {e}"));
                assert_bytes_eq(
                    &format!("two-phase read fence V {idx} round {round}"),
                    &v,
                    &pair.v_data[..4096],
                );
            }
        }
        for (idx, pair) in buffers.iter_mut().enumerate() {
            pair.k_buf
                .evict_discard()
                .unwrap_or_else(|e| panic!("two-phase discard K buffer {idx} round {round}: {e}"));
            pair.v_buf
                .evict_discard()
                .unwrap_or_else(|e| panic!("two-phase discard V buffer {idx} round {round}: {e}"));
        }
        for idx in 0..buffers.len() {
            buffers[idx]
                .k_buf
                .restore_backup()
                .unwrap_or_else(|e| panic!("two-phase restore K buffer {idx} round {round}: {e}"));
            buffers[idx]
                .v_buf
                .restore_backup()
                .unwrap_or_else(|e| panic!("two-phase restore V buffer {idx} round {round}: {e}"));
            verify_split_kv_pairs(
                &format!("two-phase round {round} after restore pair {idx}"),
                &buffers[..=idx],
            );
        }
        verify_split_kv_pairs(&format!("two-phase round {round} final"), &buffers);
    }
}

fn run_split_kv_two_phase_map_all_then_copy_restore(buffer_count: usize, rounds: usize) {
    let (_nkv, _cap, _head_dim, _elem_bytes, logical_len) = split_kv_geometry();
    let mut buffers = make_split_kv_pairs(buffer_count);
    for round in 0..rounds {
        eprintln!("map-all-copy round={round} buffers={buffer_count}");
        verify_split_kv_pairs(
            &format!("map-all-copy round {round} before backup"),
            &buffers,
        );
        for (idx, pair) in buffers.iter_mut().enumerate() {
            pair.k_buf
                .backup_mapped_to_host()
                .unwrap_or_else(|e| panic!("map-all-copy backup K buffer {idx}: {e}"));
            pair.v_buf
                .backup_mapped_to_host()
                .unwrap_or_else(|e| panic!("map-all-copy backup V buffer {idx}: {e}"));
        }
        for (idx, pair) in buffers.iter().enumerate() {
            assert_backup_eq(
                &format!("map-all-copy captured K backup {idx} round {round}"),
                &pair.k_buf,
                logical_len,
                &pair.k_data,
            );
            assert_backup_eq(
                &format!("map-all-copy captured V backup {idx} round {round}"),
                &pair.v_buf,
                logical_len,
                &pair.v_data,
            );
        }
        for (idx, pair) in buffers.iter_mut().enumerate() {
            pair.k_buf
                .evict_discard()
                .unwrap_or_else(|e| panic!("map-all-copy discard K buffer {idx}: {e}"));
            pair.v_buf
                .evict_discard()
                .unwrap_or_else(|e| panic!("map-all-copy discard V buffer {idx}: {e}"));
        }
        for (idx, pair) in buffers.iter_mut().enumerate() {
            pair.k_buf
                .map_backup_ranges_for_restore()
                .unwrap_or_else(|e| panic!("map-all-copy map K buffer {idx}: {e}"));
            pair.v_buf
                .map_backup_ranges_for_restore()
                .unwrap_or_else(|e| panic!("map-all-copy map V buffer {idx}: {e}"));
        }
        for (idx, pair) in buffers.iter_mut().enumerate() {
            pair.k_buf
                .copy_backup_to_mapped()
                .unwrap_or_else(|e| panic!("map-all-copy restore K buffer {idx}: {e}"));
            pair.v_buf
                .copy_backup_to_mapped()
                .unwrap_or_else(|e| panic!("map-all-copy restore V buffer {idx}: {e}"));
        }
        verify_split_kv_pairs(&format!("map-all-copy round {round} final"), &buffers);
    }
}

fn make_split_kv_pairs(buffer_count: usize) -> Vec<SplitKvBufferPair> {
    let nkv = 2;
    let cap = 384;
    let head_dim = 256;
    let elem_bytes = ScalarType::BF16.size_in_bytes();
    let logical_len = nkv * cap * head_dim * elem_bytes;

    let mut buffers = Vec::new();
    for idx in 0..buffer_count {
        let mut k_buf = VirtualBuffer::reserve(
            0,
            ScalarType::BF16,
            &[1, nkv, cap, head_dim],
            VirtualBacking::CpuBackup,
        )
        .unwrap_or_else(|e| panic!("reserve K buffer {idx}: {e}"));
        let mut v_buf = VirtualBuffer::reserve(
            0,
            ScalarType::BF16,
            &[1, nkv, cap, head_dim],
            VirtualBacking::CpuBackup,
        )
        .unwrap_or_else(|e| panic!("reserve V buffer {idx}: {e}"));
        k_buf
            .map_prefix_bytes(logical_len)
            .unwrap_or_else(|e| panic!("map K buffer {idx}: {e}"));
        v_buf
            .map_prefix_bytes(logical_len)
            .unwrap_or_else(|e| panic!("map V buffer {idx}: {e}"));
        eprintln!(
            "split-kv idx={idx} K={:?} V={:?} len={} reserved_k={} reserved_v={} granularity={}",
            k_buf.as_ptr(),
            v_buf.as_ptr(),
            logical_len,
            k_buf.reserved_bytes(),
            v_buf.reserved_bytes(),
            k_buf.granularity()
        );

        let k_data = pattern_bytes(logical_len)
            .into_iter()
            .map(|byte| byte.wrapping_add((idx as u8).wrapping_mul(29)))
            .collect::<Vec<_>>();
        let v_data = pattern_bytes(logical_len)
            .into_iter()
            .map(|byte| {
                byte.wrapping_add((idx as u8).wrapping_mul(29))
                    .wrapping_add(7)
            })
            .collect::<Vec<_>>();
        copy_h2d(
            0,
            k_buf.as_mut_ptr(),
            k_data.as_ptr() as *const _,
            k_data.len(),
        )
        .unwrap_or_else(|e| panic!("H2D K buffer {idx}: {e}"));
        copy_h2d(
            0,
            v_buf.as_mut_ptr(),
            v_data.as_ptr() as *const _,
            v_data.len(),
        )
        .unwrap_or_else(|e| panic!("H2D V buffer {idx}: {e}"));
        buffers.push(SplitKvBufferPair {
            k_buf,
            v_buf,
            k_data,
            v_data,
        });
    }
    sync(0).expect("sync initial split KV fills");
    verify_split_kv_pairs("initial", &buffers);
    buffers
}

fn verify_split_kv_pairs(label: &str, buffers: &[SplitKvBufferPair]) {
    let (_nkv, cap, head_dim, elem_bytes, logical_len) = split_kv_geometry();
    let prefix_len = 17;
    let src_head_stride = cap * head_dim * elem_bytes;
    let copy_bytes = prefix_len * head_dim * elem_bytes;

    for (idx, pair) in buffers.iter().enumerate() {
        let restored_k = pair
            .k_buf
            .to_host_prefix_bytes(logical_len)
            .unwrap_or_else(|e| panic!("{label}: D2H K buffer {idx}: {e}"));
        let restored_v = pair
            .v_buf
            .to_host_prefix_bytes(logical_len)
            .unwrap_or_else(|e| panic!("{label}: D2H V buffer {idx}: {e}"));
        assert_bytes_eq(
            &format!("{label}: full K buffer {idx}"),
            &restored_k,
            &pair.k_data,
        );
        assert_bytes_eq(
            &format!("{label}: full V buffer {idx}"),
            &restored_v,
            &pair.v_data,
        );
        for h in 0..2 {
            let src = h * src_head_stride;
            assert_bytes_eq(
                &format!("{label}: K buffer {idx} head {h}"),
                &restored_k[src..src + copy_bytes],
                &pair.k_data[src..src + copy_bytes],
            );
            assert_bytes_eq(
                &format!("{label}: V buffer {idx} head {h}"),
                &restored_v[src..src + copy_bytes],
                &pair.v_data[src..src + copy_bytes],
            );
        }
    }
}

fn evict_all_split_kv_pairs(buffers: &mut [SplitKvBufferPair], round: usize) {
    for (idx, pair) in buffers.iter_mut().enumerate() {
        pair.k_buf
            .evict_to_host()
            .unwrap_or_else(|e| panic!("evict K buffer {idx} round {round}: {e}"));
        pair.v_buf
            .evict_to_host()
            .unwrap_or_else(|e| panic!("evict V buffer {idx} round {round}: {e}"));
        assert_eq!(pair.k_buf.resident_bytes() + pair.v_buf.resident_bytes(), 0);
        assert_eq!(pair.k_buf.mapping_count() + pair.v_buf.mapping_count(), 0);
    }
}

fn apply_post_restore_mapping_pressure(round: usize) {
    let (_nkv, _cap, _head_dim, _elem_bytes, logical_len) = split_kv_geometry();
    let mut probes = Vec::new();
    for idx in 0..12 {
        let mut probe = VirtualBuffer::reserve(
            0,
            ScalarType::BF16,
            &[1, 2, 384, 256],
            VirtualBacking::Discard,
        )
        .unwrap_or_else(|e| panic!("reserve pressure probe {idx} round {round}: {e}"));
        probe
            .map_prefix_bytes(logical_len)
            .unwrap_or_else(|e| panic!("map pressure probe {idx} round {round}: {e}"));
        probes.push(probe);
    }
    drop(probes);
    sync(0).expect("sync post-restore mapping pressure");
}

fn run_split_kv_restore_pattern(
    buffer_count: usize,
    rounds: usize,
    pattern: SplitKvRestorePattern,
) {
    let mut buffers = make_split_kv_pairs(buffer_count);

    for round in 0..rounds {
        eprintln!("round={round} pattern={pattern:?}");
        match pattern {
            SplitKvRestorePattern::Pairwise => {
                for idx in 0..buffers.len() {
                    buffers[idx]
                        .k_buf
                        .evict_to_host()
                        .unwrap_or_else(|e| panic!("evict K buffer {idx} round {round}: {e}"));
                    buffers[idx]
                        .v_buf
                        .evict_to_host()
                        .unwrap_or_else(|e| panic!("evict V buffer {idx} round {round}: {e}"));
                    assert_eq!(
                        buffers[idx].k_buf.resident_bytes() + buffers[idx].v_buf.resident_bytes(),
                        0
                    );
                    assert_eq!(
                        buffers[idx].k_buf.mapping_count() + buffers[idx].v_buf.mapping_count(),
                        0
                    );
                    buffers[idx]
                        .k_buf
                        .restore_backup()
                        .unwrap_or_else(|e| panic!("restore K buffer {idx} round {round}: {e}"));
                    buffers[idx]
                        .v_buf
                        .restore_backup()
                        .unwrap_or_else(|e| panic!("restore V buffer {idx} round {round}: {e}"));
                    verify_split_kv_pairs(
                        &format!("pairwise round {round} after pair {idx}"),
                        &buffers,
                    );
                }
            }
            SplitKvRestorePattern::EvictAllRestorePairs
            | SplitKvRestorePattern::EvictAllRestorePairsThenMapPressure => {
                evict_all_split_kv_pairs(&mut buffers, round);
                for idx in 0..buffers.len() {
                    buffers[idx]
                        .k_buf
                        .restore_backup()
                        .unwrap_or_else(|e| panic!("restore K buffer {idx} round {round}: {e}"));
                    buffers[idx]
                        .v_buf
                        .restore_backup()
                        .unwrap_or_else(|e| panic!("restore V buffer {idx} round {round}: {e}"));
                    let label = format!("restore-pairs round {round} after pair {idx}");
                    verify_split_kv_pairs(&label, &buffers[..=idx]);
                }
                verify_split_kv_pairs(&format!("restore-pairs round {round} final"), &buffers);
                if matches!(
                    pattern,
                    SplitKvRestorePattern::EvictAllRestorePairsThenMapPressure
                ) {
                    apply_post_restore_mapping_pressure(round);
                    verify_split_kv_pairs(
                        &format!("restore-pairs round {round} after pressure"),
                        &buffers,
                    );
                }
            }
            SplitKvRestorePattern::EvictAllRestoreAllKThenAllV => {
                evict_all_split_kv_pairs(&mut buffers, round);
                for (idx, pair) in buffers.iter_mut().enumerate() {
                    pair.k_buf
                        .restore_backup()
                        .unwrap_or_else(|e| panic!("restore K buffer {idx} round {round}: {e}"));
                }
                for (idx, pair) in buffers.iter_mut().enumerate() {
                    pair.v_buf
                        .restore_backup()
                        .unwrap_or_else(|e| panic!("restore V buffer {idx} round {round}: {e}"));
                }
                verify_split_kv_pairs(&format!("K-then-V round {round} final"), &buffers);
            }
            SplitKvRestorePattern::EvictAllRestoreReversePairs => {
                evict_all_split_kv_pairs(&mut buffers, round);
                for idx in (0..buffers.len()).rev() {
                    buffers[idx]
                        .k_buf
                        .restore_backup()
                        .unwrap_or_else(|e| panic!("restore K buffer {idx} round {round}: {e}"));
                    buffers[idx]
                        .v_buf
                        .restore_backup()
                        .unwrap_or_else(|e| panic!("restore V buffer {idx} round {round}: {e}"));
                    let label = format!("reverse-pairs round {round} after pair {idx}");
                    verify_split_kv_pairs(&label, &buffers[idx..]);
                }
                verify_split_kv_pairs(&format!("reverse-pairs round {round} final"), &buffers);
            }
        }
    }
}

#[test]
fn vmm_sparse_cpu_backup_restore_round_trip() {
    set_backend(Backend::Hip);
    if !vmm_is_supported(Backend::Hip, 0) {
        eprintln!("skip: HIP VMM unsupported on this device/runtime");
        return;
    }

    let mut buf = VirtualBuffer::reserve(0, ScalarType::U8, &[1 << 20], VirtualBacking::CpuBackup)
        .expect("reserve virtual buffer");
    let first_offset = 64 * 1024;
    let second_offset = 768 * 1024;
    let len = 16 * 1024;
    let first = pattern_bytes(len);
    let second = pattern_bytes(len)
        .into_iter()
        .map(|byte| byte.wrapping_add(37))
        .collect::<Vec<_>>();

    buf.map_range_bytes(first_offset, len)
        .expect("map first island");
    buf.map_range_bytes(second_offset, len)
        .expect("map second island");
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

    buf.backup_mapped_to_host().expect("backup sparse islands");
    buf.unmap_all().expect("unmap sparse islands");
    assert_eq!(buf.resident_bytes(), 0);
    assert_eq!(buf.mapping_count(), 0);
    buf.restore_backup().expect("restore sparse islands");
    sync(0).expect("sync sparse restore");

    assert_eq!(
        buf.to_host_range_bytes(first_offset, len)
            .expect("D2H first island"),
        first
    );
    assert_eq!(
        buf.to_host_range_bytes(second_offset, len)
            .expect("D2H second island"),
        second
    );
}
