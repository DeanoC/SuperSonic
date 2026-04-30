//! Round-trip integration tests for the `BufferKind` / `BufferPolicy`
//! plumbing.
//!
//! These exercise the same access pattern used by
//! `decode_engine::load_kv_shadow_for_state_static`:
//!
//!   1. allocate a `Scratch` buffer via `from_host_bytes_with_kind`
//!      (writes via H2D copy through the policy-resolved allocator)
//!   2. allocate a `Persistent` destination via `zeros_with_kind`
//!   3. `copy_d2d` between them
//!   4. read the destination back and bit-compare against the input
//!   5. drop both buffers — the two distinct allocator kinds get freed
//!      via their matching free path (`hipFree` for Default,
//!      `hipHostFree(host_ptr)` for `HostMapped`).
//!
//! Run twice: once under the `gfx1150` policy
//! (`{Persistent: Default, Scratch: HostMapped}`) so the host-mapped
//! branch in `ops::alloc` actually executes, and once under the
//! all-`Default` policy used by `gfx1100` / `sm86` / unmeasured arches
//! so the `Default` branch round-trips for every kind. Functional
//! correctness is the contract — perf is measured separately
//! (`tests/gfx1150/membench_l2_bypass.hip`).

#![cfg(supersonic_backend_hip)]

use gpu_hal::{
    copy_d2d, set_backend, set_buffer_policy, sync, AllocStrategy, Backend, BufferKind,
    BufferPolicy, GpuBuffer, ScalarType,
};

fn host_bf16_pattern(n: usize) -> Vec<u8> {
    // Distinct-per-element BF16 bit pattern; bit-exact comparison only
    // works because we never go through any cast.
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let bits = (0x3f00u16).wrapping_add(i as u16);
        out.extend_from_slice(&bits.to_le_bytes());
    }
    out
}

fn round_trip_under_policy(policy: BufferPolicy, src_kind: BufferKind, dst_kind: BufferKind) {
    set_backend(Backend::Hip);
    set_buffer_policy(policy);

    let ordinal = 0usize;
    let n_elems = 4096;
    let host = host_bf16_pattern(n_elems);

    let src =
        GpuBuffer::from_host_bytes_with_kind(ordinal, ScalarType::BF16, &[n_elems], &host, src_kind)
            .expect("alloc + H2D src");
    let mut dst = GpuBuffer::zeros_with_kind(ordinal, ScalarType::BF16, &[n_elems], dst_kind)
        .expect("alloc dst");

    copy_d2d(ordinal, dst.as_mut_ptr(), src.as_ptr(), src.len_bytes()).expect("copy_d2d");
    sync(ordinal).expect("sync after copy_d2d");

    let copied = dst.to_host_bytes().expect("D2H readback");
    assert_eq!(copied, host, "round-trip bytes mismatch");
    // Drop sequence: dst then src, exercises both allocator-kind free paths
    // back-to-back when the policy maps them to different mechanisms.
}

#[test]
fn gfx1150_policy_scratch_round_trip() {
    // Mirrors the gfx1150 entry in registry::ArchProfile::for_arch:
    // Persistent stays on the classic device allocator, Scratch opts
    // into hipHostMalloc(MAPPED) + hipHostGetDevicePointer.
    let policy = BufferPolicy {
        persistent: AllocStrategy::Default,
        scratch: AllocStrategy::HostMapped,
    };
    round_trip_under_policy(policy, BufferKind::Scratch, BufferKind::Persistent);
}

#[test]
fn gfx1150_policy_persistent_round_trip() {
    // Same policy, but route the source through Persistent — confirms
    // the Default branch still works when the policy table has a
    // HostMapped entry available for Scratch.
    let policy = BufferPolicy {
        persistent: AllocStrategy::Default,
        scratch: AllocStrategy::HostMapped,
    };
    round_trip_under_policy(policy, BufferKind::Persistent, BufferKind::Persistent);
}

#[test]
fn discrete_policy_round_trip_for_both_kinds() {
    // Mirrors gfx1100 / sm86 / unmeasured arches: every kind maps to
    // Default. Both branches in the policy lookup take the same path,
    // and Scratch is a no-op opt-in.
    let policy = BufferPolicy::all_default();
    round_trip_under_policy(policy, BufferKind::Scratch, BufferKind::Scratch);
    round_trip_under_policy(policy, BufferKind::Persistent, BufferKind::Persistent);
}
