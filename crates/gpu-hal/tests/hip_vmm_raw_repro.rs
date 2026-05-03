#![cfg(supersonic_backend_hip)]

// Standalone HIP VMM repro for AMD/runtime triage.
//
// This test intentionally duplicates the small subset of HIP FFI it needs
// instead of using gpu-hal::VirtualBuffer. The failing tests prove the remap
// corruption can be reproduced with direct HIP calls only.

use std::ffi::{c_int, c_uint, c_ulonglong, c_void};
use std::ptr::NonNull;

const HIP_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const HIP_MEMCPY_DEVICE_TO_HOST: c_int = 2;
const HIP_MEM_ACCESS_FLAGS_PROT_READ_WRITE: c_uint = 3;
const HIP_MEM_ALLOCATION_TYPE_PINNED: c_uint = 1;
const HIP_MEM_HANDLE_TYPE_NONE: c_uint = 0;
const HIP_MEM_LOCATION_TYPE_DEVICE: c_uint = 1;
const HIP_MEM_LOCATION_TYPE_HOST: c_uint = 2;
const HIP_MEM_ALLOCATION_GRANULARITY_RECOMMENDED: c_uint = 1;

type HipMemGenericAllocationHandle = *mut c_void;

#[repr(C)]
#[derive(Clone, Copy)]
struct HipMemLocation {
    type_: c_uint,
    id: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct HipMemAccessDesc {
    location: HipMemLocation,
    flags: c_uint,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct HipMemAllocationPropAllocFlags {
    compression_type: u8,
    gpu_direct_rdma_capable: u8,
    usage: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct HipMemAllocationProp {
    type_: c_uint,
    requested_handle_type: c_uint,
    location: HipMemLocation,
    win32_handle_meta_data: *mut c_void,
    alloc_flags: HipMemAllocationPropAllocFlags,
}

#[link(name = "amdhip64")]
unsafe extern "C" {
    fn hipGetDevice(device: *mut c_int) -> c_int;
    fn hipDeviceSynchronize() -> c_int;
    fn hipMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: c_int) -> c_int;
    fn hipMemset(dst: *mut c_void, value: c_int, size: usize) -> c_int;
    fn hipMemAddressReserve(
        ptr: *mut *mut c_void,
        size: usize,
        alignment: usize,
        addr: *mut c_void,
        flags: c_ulonglong,
    ) -> c_int;
    fn hipMemAddressFree(dev_ptr: *mut c_void, size: usize) -> c_int;
    fn hipMemCreate(
        handle: *mut HipMemGenericAllocationHandle,
        size: usize,
        prop: *const HipMemAllocationProp,
        flags: c_ulonglong,
    ) -> c_int;
    fn hipMemRelease(handle: HipMemGenericAllocationHandle) -> c_int;
    fn hipMemGetAllocationGranularity(
        granularity: *mut usize,
        prop: *const HipMemAllocationProp,
        option: c_uint,
    ) -> c_int;
    fn hipMemMap(
        ptr: *mut c_void,
        size: usize,
        offset: usize,
        handle: HipMemGenericAllocationHandle,
        flags: c_ulonglong,
    ) -> c_int;
    fn hipMemUnmap(ptr: *mut c_void, size: usize) -> c_int;
    fn hipMemSetAccess(
        ptr: *mut c_void,
        size: usize,
        desc: *const HipMemAccessDesc,
        count: usize,
    ) -> c_int;
}

#[derive(Clone, Copy, Debug)]
enum HandleLifetime {
    ReleaseAfterMap,
    RetainUntilUnmap,
}

#[derive(Clone, Copy, Debug)]
enum AccessScope {
    DeviceOnly,
    DeviceAndHost,
}

struct RawHipVmm {
    ordinal: c_int,
    ptr: NonNull<c_void>,
    len: usize,
    alignment: usize,
    handle: Option<HipMemGenericAllocationHandle>,
    access_scope: AccessScope,
}

impl Drop for RawHipVmm {
    fn drop(&mut self) {
        unsafe {
            let _ = hipMemUnmap(self.ptr.as_ptr(), self.len);
            if let Some(handle) = self.handle.take() {
                let _ = hipMemRelease(handle);
            }
            let _ = hipMemAddressFree(self.ptr.as_ptr(), self.len);
        }
    }
}

impl RawHipVmm {
    fn reserve_map(
        ordinal: c_int,
        len: usize,
        lifetime: HandleLifetime,
        access_scope: AccessScope,
    ) -> Self {
        let prop = allocation_prop(ordinal);
        let granularity = allocation_granularity(&prop);
        let len = align_up(len, granularity);
        let mut ptr = std::ptr::null_mut();
        hip_ok(
            unsafe { hipMemAddressReserve(&mut ptr, len, granularity, std::ptr::null_mut(), 0) },
            "hipMemAddressReserve",
        );
        let ptr = NonNull::new(ptr).expect("hipMemAddressReserve returned null");
        let mut buf = Self {
            ordinal,
            ptr,
            len,
            alignment: granularity,
            handle: None,
            access_scope,
        };
        buf.map(lifetime);
        buf
    }

    fn map(&mut self, lifetime: HandleLifetime) {
        let prop = allocation_prop(self.ordinal);
        let mut handle = std::ptr::null_mut();
        hip_ok(
            unsafe { hipMemCreate(&mut handle, self.len, &prop, 0) },
            "hipMemCreate",
        );
        hip_ok(
            unsafe { hipMemMap(self.ptr.as_ptr(), self.len, 0, handle, 0) },
            "hipMemMap",
        );
        let mut access = [
            HipMemAccessDesc {
                location: HipMemLocation {
                    type_: HIP_MEM_LOCATION_TYPE_DEVICE,
                    id: self.ordinal,
                },
                flags: HIP_MEM_ACCESS_FLAGS_PROT_READ_WRITE,
            },
            HipMemAccessDesc {
                location: HipMemLocation {
                    type_: HIP_MEM_LOCATION_TYPE_HOST,
                    id: 0,
                },
                flags: HIP_MEM_ACCESS_FLAGS_PROT_READ_WRITE,
            },
        ];
        let access_count = match self.access_scope {
            AccessScope::DeviceOnly => 1,
            AccessScope::DeviceAndHost => 2,
        };
        hip_ok(
            unsafe {
                hipMemSetAccess(
                    self.ptr.as_ptr(),
                    self.len,
                    access.as_mut_ptr(),
                    access_count,
                )
            },
            "hipMemSetAccess",
        );
        hip_ok(
            unsafe { hipMemset(self.ptr.as_ptr(), 0, self.len) },
            "hipMemset zero after map",
        );
        sync();
        match lifetime {
            HandleLifetime::ReleaseAfterMap => {
                hip_ok(unsafe { hipMemRelease(handle) }, "hipMemRelease after map");
                self.handle = None;
            }
            HandleLifetime::RetainUntilUnmap => {
                self.handle = Some(handle);
            }
        }
    }

    fn unmap(&mut self) {
        hip_ok(
            unsafe { hipMemUnmap(self.ptr.as_ptr(), self.len) },
            "hipMemUnmap",
        );
        if let Some(handle) = self.handle.take() {
            hip_ok(unsafe { hipMemRelease(handle) }, "hipMemRelease at unmap");
        }
        sync();
    }

    fn remap(&mut self, lifetime: HandleLifetime) {
        self.map(lifetime);
    }

    fn recycle_address_reservation(&mut self) {
        let old = self.ptr.as_ptr();
        hip_ok(
            unsafe { hipMemAddressFree(old, self.len) },
            "hipMemAddressFree recycle reservation",
        );
        let mut new_ptr = std::ptr::null_mut();
        hip_ok(
            unsafe { hipMemAddressReserve(&mut new_ptr, self.len, self.alignment, old, 0) },
            "hipMemAddressReserve recycle reservation",
        );
        assert_eq!(
            new_ptr, old,
            "HIP did not honor exact-address reservation during recycle"
        );
        self.ptr = NonNull::new(new_ptr).expect("recycled reservation returned null");
    }

    fn fill(&self, data: &[u8], label: &str) {
        assert!(data.len() <= self.len);
        hip_ok(
            unsafe {
                hipMemcpy(
                    self.ptr.as_ptr(),
                    data.as_ptr() as *const c_void,
                    data.len(),
                    HIP_MEMCPY_HOST_TO_DEVICE,
                )
            },
            label,
        );
    }

    fn read(&self, len: usize, label: &str) -> Vec<u8> {
        assert!(len <= self.len);
        sync();
        let mut out = vec![0u8; len];
        hip_ok(
            unsafe {
                hipMemcpy(
                    out.as_mut_ptr() as *mut c_void,
                    self.ptr.as_ptr(),
                    len,
                    HIP_MEMCPY_DEVICE_TO_HOST,
                )
            },
            label,
        );
        sync();
        out
    }
}

struct RawPair {
    k: RawHipVmm,
    v: RawHipVmm,
    k_data: Vec<u8>,
    v_data: Vec<u8>,
}

fn current_device() -> c_int {
    let mut device = 0;
    hip_ok(unsafe { hipGetDevice(&mut device) }, "hipGetDevice");
    device
}

fn allocation_prop(ordinal: c_int) -> HipMemAllocationProp {
    HipMemAllocationProp {
        type_: HIP_MEM_ALLOCATION_TYPE_PINNED,
        requested_handle_type: HIP_MEM_HANDLE_TYPE_NONE,
        location: HipMemLocation {
            type_: HIP_MEM_LOCATION_TYPE_DEVICE,
            id: ordinal,
        },
        win32_handle_meta_data: std::ptr::null_mut(),
        alloc_flags: HipMemAllocationPropAllocFlags {
            compression_type: 0,
            gpu_direct_rdma_capable: 0,
            usage: 0,
        },
    }
}

fn allocation_granularity(prop: &HipMemAllocationProp) -> usize {
    let mut granularity = 0usize;
    hip_ok(
        unsafe {
            hipMemGetAllocationGranularity(
                &mut granularity,
                prop,
                HIP_MEM_ALLOCATION_GRANULARITY_RECOMMENDED,
            )
        },
        "hipMemGetAllocationGranularity",
    );
    assert!(granularity > 0);
    granularity
}

fn hip_ok(status: c_int, op: &str) {
    assert_eq!(status, 0, "{op} failed with HIP status {status}");
}

fn sync() {
    hip_ok(unsafe { hipDeviceSynchronize() }, "hipDeviceSynchronize");
}

fn align_up(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}

fn pattern_bytes(n: usize, salt: u8) -> Vec<u8> {
    (0..n)
        .map(|i| {
            (i as u8)
                .wrapping_mul(17)
                .wrapping_add(3)
                .wrapping_add(salt)
        })
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
    let pos = diff.unwrap_or(0);
    let start = pos.saturating_sub(8);
    let end = (pos + 24).min(got.len()).min(expected.len());
    assert!(
        diff.is_none(),
        "{label} mismatch first_diff={diff:?} got_prefix={:?} expected_prefix={:?} got_at_diff={:?} expected_at_diff={:?}",
        &got[..got.len().min(16)],
        &expected[..expected.len().min(16)],
        &got[start..end],
        &expected[start..end]
    );
}

fn make_pairs(
    pair_count: usize,
    lifetime: HandleLifetime,
    access_scope: AccessScope,
    logical_len: usize,
) -> Vec<RawPair> {
    make_pairs_with_reservation(pair_count, lifetime, access_scope, logical_len, logical_len)
}

fn make_pairs_with_reservation(
    pair_count: usize,
    lifetime: HandleLifetime,
    access_scope: AccessScope,
    logical_len: usize,
    reservation_len: usize,
) -> Vec<RawPair> {
    let ordinal = current_device();
    let mut pairs = Vec::new();
    for idx in 0..pair_count {
        let k = RawHipVmm::reserve_map(ordinal, reservation_len, lifetime, access_scope);
        let v = RawHipVmm::reserve_map(ordinal, reservation_len, lifetime, access_scope);
        eprintln!(
            "raw hip pair={idx} K={:?} V={:?} len={} lifetime={lifetime:?} access_scope={access_scope:?}",
            k.ptr, v.ptr, k.len
        );
        let k_data = pattern_bytes(logical_len, (idx as u8).wrapping_mul(29));
        let v_data = pattern_bytes(logical_len, (idx as u8).wrapping_mul(29).wrapping_add(7));
        k.fill(&k_data, "hipMemcpy H2D K fill");
        v.fill(&v_data, "hipMemcpy H2D V fill");
        pairs.push(RawPair {
            k,
            v,
            k_data,
            v_data,
        });
    }
    sync();
    verify_pairs("initial", &pairs, logical_len);
    pairs
}

fn verify_pairs(label: &str, pairs: &[RawPair], logical_len: usize) {
    for (idx, pair) in pairs.iter().enumerate() {
        let k = pair.k.read(logical_len, "hipMemcpy D2H K verify");
        let v = pair.v.read(logical_len, "hipMemcpy D2H V verify");
        assert_bytes_eq(&format!("{label}: K pair {idx}"), &k, &pair.k_data);
        assert_bytes_eq(&format!("{label}: V pair {idx}"), &v, &pair.v_data);
    }
}

fn raw_pairwise_remap_restore(pair_count: usize, rounds: usize, lifetime: HandleLifetime) {
    raw_pairwise_remap_restore_impl(pair_count, rounds, lifetime, AccessScope::DeviceOnly, false);
}

fn raw_pairwise_sequential_remap_restore(
    pair_count: usize,
    rounds: usize,
    lifetime: HandleLifetime,
) {
    raw_pairwise_remap_restore_impl(pair_count, rounds, lifetime, AccessScope::DeviceOnly, true);
}

fn raw_pairwise_sequential_remap_restore_with_access(
    pair_count: usize,
    rounds: usize,
    lifetime: HandleLifetime,
    access_scope: AccessScope,
) {
    raw_pairwise_remap_restore_impl(pair_count, rounds, lifetime, access_scope, true);
}

fn raw_pairwise_remap_restore_impl(
    pair_count: usize,
    rounds: usize,
    lifetime: HandleLifetime,
    access_scope: AccessScope,
    sequential_restore: bool,
) {
    let logical_len = 2 * 384 * 256 * 2;
    let mut pairs = make_pairs(pair_count, lifetime, access_scope, logical_len);

    for round in 0..rounds {
        eprintln!(
            "raw hip round={round} pair_count={pair_count} lifetime={lifetime:?} access_scope={access_scope:?} sequential_restore={sequential_restore}"
        );
        for idx in 0..pairs.len() {
            let k_backup = pairs[idx].k.read(logical_len, "hipMemcpy D2H K backup");
            let v_backup = pairs[idx].v.read(logical_len, "hipMemcpy D2H V backup");
            assert_bytes_eq(
                &format!("captured K backup pair {idx} round {round}"),
                &k_backup,
                &pairs[idx].k_data,
            );
            assert_bytes_eq(
                &format!("captured V backup pair {idx} round {round}"),
                &v_backup,
                &pairs[idx].v_data,
            );
            pairs[idx].k.unmap();
            pairs[idx].v.unmap();
            if sequential_restore {
                pairs[idx].k.remap(lifetime);
                pairs[idx].k.fill(&k_backup, "hipMemcpy H2D K restore");
                sync();
                let k_check = pairs[idx]
                    .k
                    .read(logical_len, "hipMemcpy D2H K post-restore");
                assert_bytes_eq(
                    &format!("post-restore K verify pair {idx} round {round}"),
                    &k_check,
                    &pairs[idx].k_data,
                );
                pairs[idx].v.remap(lifetime);
                pairs[idx].v.fill(&v_backup, "hipMemcpy H2D V restore");
                sync();
                let v_check = pairs[idx]
                    .v
                    .read(logical_len, "hipMemcpy D2H V post-restore");
                assert_bytes_eq(
                    &format!("post-restore V verify pair {idx} round {round}"),
                    &v_check,
                    &pairs[idx].v_data,
                );
            } else {
                pairs[idx].k.remap(lifetime);
                pairs[idx].v.remap(lifetime);
                pairs[idx].k.fill(&k_backup, "hipMemcpy H2D K restore");
                pairs[idx].v.fill(&v_backup, "hipMemcpy H2D V restore");
                sync();
            }
            verify_pairs(
                &format!("after restore pair {idx} round {round}"),
                &pairs[..=idx],
                logical_len,
            );
        }
        verify_pairs(&format!("after round {round}"), &pairs, logical_len);
    }
}

fn raw_all_pairs_map_then_copy_restore(
    pair_count: usize,
    rounds: usize,
    recycle_va: bool,
    reservation_len: Option<usize>,
) {
    let logical_len = 2 * 384 * 256 * 2;
    let mut pairs = make_pairs_with_reservation(
        pair_count,
        HandleLifetime::ReleaseAfterMap,
        AccessScope::DeviceOnly,
        logical_len,
        reservation_len.unwrap_or(logical_len),
    );

    for round in 0..rounds {
        eprintln!(
            "raw hip all-map-copy round={round} pair_count={pair_count} recycle_va={recycle_va}"
        );
        let mut backups = Vec::with_capacity(pairs.len());
        for (idx, pair) in pairs.iter().enumerate() {
            let k_backup = pair.k.read(logical_len, "hipMemcpy D2H K backup");
            let v_backup = pair.v.read(logical_len, "hipMemcpy D2H V backup");
            assert_bytes_eq(
                &format!("captured K backup pair {idx} round {round}"),
                &k_backup,
                &pair.k_data,
            );
            assert_bytes_eq(
                &format!("captured V backup pair {idx} round {round}"),
                &v_backup,
                &pair.v_data,
            );
            backups.push((k_backup, v_backup));
        }
        for pair in &mut pairs {
            pair.k.unmap();
            pair.v.unmap();
        }
        if recycle_va {
            for pair in &mut pairs {
                pair.k.recycle_address_reservation();
                pair.v.recycle_address_reservation();
            }
        }
        for pair in &mut pairs {
            pair.k.remap(HandleLifetime::ReleaseAfterMap);
            pair.v.remap(HandleLifetime::ReleaseAfterMap);
        }
        for (idx, pair) in pairs.iter().enumerate() {
            pair.k.fill(&backups[idx].0, "hipMemcpy H2D K restore");
            pair.v.fill(&backups[idx].1, "hipMemcpy H2D V restore");
        }
        sync();
        verify_pairs(
            &format!("after all-map-copy round {round}"),
            &pairs,
            logical_len,
        );
    }
}

#[test]
#[ignore = "Raw HIP VMM control: one split K/V pair remaps and restores once"]
fn raw_hip_vmm_single_pair_one_round_remap_restore_control() {
    raw_pairwise_remap_restore(1, 1, HandleLifetime::ReleaseAfterMap);
}

#[test]
#[ignore = "Raw HIP VMM repro: one split K/V pair corrupts after repeated remap/restore"]
fn raw_hip_vmm_single_pair_repeated_remap_restore_repro() {
    raw_pairwise_remap_restore(1, 3, HandleLifetime::ReleaseAfterMap);
}

#[test]
#[ignore = "Raw HIP VMM repro: release physical allocation handles immediately after map"]
fn raw_hip_vmm_two_pair_remap_restore_release_after_map_repro() {
    raw_pairwise_remap_restore(2, 2, HandleLifetime::ReleaseAfterMap);
}

#[test]
#[ignore = "Raw HIP VMM repro: sub-2 MiB release-after-map sequential restore corrupts"]
fn raw_hip_vmm_two_pair_sequential_restore_release_after_map_repro() {
    raw_pairwise_sequential_remap_restore(2, 2, HandleLifetime::ReleaseAfterMap);
}

#[test]
#[ignore = "Raw HIP VMM check: host access descriptor does not fix sub-2 MiB sequential restore"]
fn raw_hip_vmm_two_pair_sequential_restore_device_and_host_access_check() {
    raw_pairwise_sequential_remap_restore_with_access(
        2,
        2,
        HandleLifetime::ReleaseAfterMap,
        AccessScope::DeviceAndHost,
    );
}

#[test]
#[ignore = "Raw HIP VMM check: retaining physical handles until unmap still corrupts sub-2 MiB restore"]
fn raw_hip_vmm_two_pair_remap_restore_retain_until_unmap_check() {
    raw_pairwise_remap_restore(2, 2, HandleLifetime::RetainUntilUnmap);
}

#[test]
#[ignore = "Raw HIP VMM repro: sub-2 MiB remap of every split K/V range corrupts host restore"]
fn raw_hip_vmm_all_pairs_map_then_copy_restore_repro() {
    raw_all_pairs_map_then_copy_restore(6, 1, false, None);
}

#[test]
#[ignore = "Raw HIP VMM check: exact-address VA recycle before remap/restore"]
fn raw_hip_vmm_all_pairs_recycle_va_map_then_copy_restore_check() {
    raw_all_pairs_map_then_copy_restore(6, 1, true, None);
}

#[test]
#[ignore = "Raw HIP VMM control: 2 MiB virtual/physical mappings survive remap/restore"]
fn raw_hip_vmm_all_pairs_two_mib_map_then_copy_restore_control() {
    raw_all_pairs_map_then_copy_restore(6, 1, false, Some(2 * 1024 * 1024));
}
