use super::*;
use gpu_hal::{Backend, VirtualAllocationRole};
use model_store::manifest::{LayoutTag, Manifest, TensorMeta, FORMAT_VERSION};
use std::sync::Mutex;

static VMM_BACKEND_TEST_LOCK: Mutex<()> = Mutex::new(());

struct BackendRestore(Backend);

impl Drop for BackendRestore {
    fn drop(&mut self) {
        gpu_hal::set_backend(self.0);
    }
}

fn synthetic_store(expert_count: usize, expert_bytes: usize) -> tempfile::TempDir {
    let tmp = tempfile::tempdir().expect("tempdir");
    let bake_dir = tmp.path();
    let total_bytes = expert_count * expert_bytes;
    let mut weights = vec![0u8; total_bytes];
    for expert in 0..expert_count {
        let fill = (expert as u8).wrapping_add(1);
        weights[expert * expert_bytes..(expert + 1) * expert_bytes].fill(fill);
    }
    std::fs::write(model_store::weights_bin_path(bake_dir), &weights).expect("write weights.bin");
    let manifest = Manifest {
        format_version: FORMAT_VERSION,
        converter_version: 1,
        model_family: "test-qwen36-moe".to_string(),
        quant_profile: None,
        source_format: None,
        source_quant: None,
        tensors: vec![TensorMeta {
            name: "model.layers.0.mlp.experts.gate_up_proj".to_string(),
            shape: vec![total_bytes],
            dtype: "u8".to_string(),
            layout: LayoutTag::Int4Quantized,
            offset: 0,
            byte_len: total_bytes as u64,
        }],
    };
    std::fs::write(
        model_store::manifest_path(bake_dir),
        serde_json::to_string(&manifest).expect("serialize manifest"),
    )
    .expect("write manifest");
    tmp
}

fn vmm_backends() -> [Backend; 2] {
    [Backend::Hip, Backend::Cuda]
}

fn with_supported_vmm_backend(test_name: &str, mut f: impl FnMut(Backend)) {
    let _lock = VMM_BACKEND_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let _restore_backend = BackendRestore(gpu_hal::current_backend());

    for backend in vmm_backends() {
        gpu_hal::set_backend(backend);
        if !gpu_hal::vmm_is_supported(backend, 0) {
            eprintln!("skip: {backend} VMM unsupported on this device/runtime for {test_name}");
            continue;
        }
        f(backend);
    }
}

#[test]
fn reserves_expert_slab_without_resident_pages() {
    with_supported_vmm_backend("reserves_expert_slab_without_resident_pages", |_backend| {
        let tmp = synthetic_store(4, 4096);
        let store = BakedStore::open(tmp.path()).expect("open synthetic store");
        let mut manager =
            MoeExpertResidencyManager::new(0, MoeExpertResidencyConfig::new(2).unwrap());
        let reservation = manager
            .register_tensor(
                &store,
                0,
                MoeExpertProjection::GateUp,
                "model.layers.0.mlp.experts.gate_up_proj",
                4,
            )
            .expect("register tensor");

        assert_eq!(reservation.expert_count, 4);
        assert_eq!(reservation.expert_bytes, 4096);
        let arena_stats = manager.arena().stats();
        assert_eq!(arena_stats.allocations, 1);
        assert_eq!(arena_stats.logical_bytes, 16 * 1024);
        assert_eq!(arena_stats.logical_resident_bytes, 0);
        assert_eq!(arena_stats.mapping_count, 0);
    });
}

#[test]
fn shared_page_reuses_backing_for_neighbor_slice() {
    with_supported_vmm_backend(
        "shared_page_reuses_backing_for_neighbor_slice",
        |_backend| {
            let tmp = synthetic_store(4, 4096);
            let store = BakedStore::open(tmp.path()).expect("open synthetic store");
            let mut manager =
                MoeExpertResidencyManager::new(0, MoeExpertResidencyConfig::new(1).unwrap());
            manager
                .register_tensor(
                    &store,
                    0,
                    MoeExpertProjection::GateUp,
                    "model.layers.0.mlp.experts.gate_up_proj",
                    4,
                )
                .expect("register tensor");

            let e0 = MoeExpertKey {
                layer_idx: 0,
                expert_idx: 0,
                projection: MoeExpertProjection::GateUp,
            };
            let e1 = MoeExpertKey {
                layer_idx: 0,
                expert_idx: 1,
                projection: MoeExpertProjection::GateUp,
            };
            manager.ensure_resident(&store, e0).expect("load expert 0");
            assert!(manager.is_resident(e0));
            assert_eq!(manager.stats().resident_slices, 1);
            assert_eq!(manager.stats().resident_pages, 1);
            assert_eq!(manager.stats().page_misses, 1);

            manager.ensure_resident(&store, e1).expect("load expert 1");
            assert!(manager.is_resident(e0));
            assert!(manager.is_resident(e1));
            assert_eq!(manager.stats().resident_slices, 2);
            assert_eq!(manager.stats().resident_pages, 1);
            assert!(manager.stats().page_backed_slices >= 2);
            assert_eq!(manager.stats().misses, 2);
            assert_eq!(manager.stats().page_hits, 1);
            assert_eq!(manager.stats().page_misses, 1);
            assert_eq!(manager.stats().evicted_pages, 0);
            assert_eq!(manager.stats().evicted_slices, 0);

            let allocation_id = manager
                .resident_weight(0, MoeExpertProjection::GateUp)
                .expect("resident weight")
                .allocation_id()
                .expect("virtual allocation");
            let bytes = manager
                .arena()
                .allocation(allocation_id)
                .expect("allocation")
                .buffer()
                .to_host_range_bytes(4096, 4096)
                .expect("read expert 1");
            assert!(bytes.iter().all(|b| *b == 2));
        },
    );
}

#[test]
fn lru_eviction_invalidates_whole_pages() {
    with_supported_vmm_backend("lru_eviction_invalidates_whole_pages", |_backend| {
        let probe_tmp = synthetic_store(1, 4096);
        let probe_store = BakedStore::open(probe_tmp.path()).expect("open probe store");
        let mut probe =
            MoeExpertResidencyManager::new(0, MoeExpertResidencyConfig::new(1).unwrap());
        probe
            .register_tensor(
                &probe_store,
                0,
                MoeExpertProjection::GateUp,
                "model.layers.0.mlp.experts.gate_up_proj",
                1,
            )
            .expect("register probe tensor");
        let expert_bytes = probe.tensors[0].page_bytes;

        let tmp = synthetic_store(3, expert_bytes);
        let store = BakedStore::open(tmp.path()).expect("open synthetic store");
        let mut manager =
            MoeExpertResidencyManager::new(0, MoeExpertResidencyConfig::new(1).unwrap());
        manager
            .register_tensor(
                &store,
                0,
                MoeExpertProjection::GateUp,
                "model.layers.0.mlp.experts.gate_up_proj",
                3,
            )
            .expect("register tensor");

        let e0 = MoeExpertKey {
            layer_idx: 0,
            expert_idx: 0,
            projection: MoeExpertProjection::GateUp,
        };
        let e1 = MoeExpertKey {
            layer_idx: 0,
            expert_idx: 1,
            projection: MoeExpertProjection::GateUp,
        };

        manager.ensure_resident(&store, e0).expect("load expert 0");
        assert!(manager.is_resident(e0));
        assert_eq!(manager.stats().resident_pages, 1);

        manager.ensure_resident(&store, e1).expect("load expert 1");
        assert!(!manager.is_resident(e0));
        assert!(manager.is_resident(e1));
        assert_eq!(manager.stats().resident_pages, 1);
        assert_eq!(manager.stats().page_misses, 2);
        assert!(manager.stats().evicted_pages >= 1);
        assert!(manager.stats().evicted_slices >= 1);

        let allocation_id = manager
            .resident_weight(0, MoeExpertProjection::GateUp)
            .expect("resident weight")
            .allocation_id()
            .expect("virtual allocation");
        let bytes = manager
            .arena()
            .allocation(allocation_id)
            .expect("allocation")
            .buffer()
            .to_host_range_bytes(expert_bytes, 4096)
            .expect("read expert 1");
        assert!(bytes.iter().all(|b| *b == 2));
    });
}

#[test]
fn protected_pages_are_evicted_after_unprotected_pages() {
    with_supported_vmm_backend(
        "protected_pages_are_evicted_after_unprotected_pages",
        |_backend| {
            let probe_tmp = synthetic_store(1, 4096);
            let probe_store = BakedStore::open(probe_tmp.path()).expect("open probe store");
            let mut probe =
                MoeExpertResidencyManager::new(0, MoeExpertResidencyConfig::new(1).unwrap());
            probe
                .register_tensor(
                    &probe_store,
                    0,
                    MoeExpertProjection::GateUp,
                    "model.layers.0.mlp.experts.gate_up_proj",
                    1,
                )
                .expect("register probe tensor");
            let expert_bytes = probe.tensors[0].page_bytes;

            let tmp = synthetic_store(3, expert_bytes);
            let store = BakedStore::open(tmp.path()).expect("open synthetic store");
            let mut manager =
                MoeExpertResidencyManager::new(0, MoeExpertResidencyConfig::new(2).unwrap());
            manager.set_max_protected_pages(1);
            manager
                .register_tensor(
                    &store,
                    0,
                    MoeExpertProjection::GateUp,
                    "model.layers.0.mlp.experts.gate_up_proj",
                    3,
                )
                .expect("register tensor");

            let e0 = MoeExpertKey {
                layer_idx: 0,
                expert_idx: 0,
                projection: MoeExpertProjection::GateUp,
            };
            let e1 = MoeExpertKey {
                layer_idx: 0,
                expert_idx: 1,
                projection: MoeExpertProjection::GateUp,
            };
            let e2 = MoeExpertKey {
                layer_idx: 0,
                expert_idx: 2,
                projection: MoeExpertProjection::GateUp,
            };

            manager.ensure_resident(&store, e0).expect("load expert 0");
            manager.protect_resident(e0).expect("protect expert 0");
            manager.ensure_resident(&store, e1).expect("load expert 1");
            manager.ensure_resident(&store, e2).expect("load expert 2");

            assert!(manager.is_resident(e0));
            assert!(!manager.is_resident(e1));
            assert!(manager.is_resident(e2));
            assert_eq!(manager.stats().protected_pages, 1);
            assert_eq!(manager.stats().protect_hits, 1);
            assert!(manager.stats().evicted_pages >= 1);
        },
    );
}

#[test]
fn prefetch_does_not_evict_when_page_budget_is_full() {
    with_supported_vmm_backend(
        "prefetch_does_not_evict_when_page_budget_is_full",
        |_backend| {
            let probe_tmp = synthetic_store(1, 4096);
            let probe_store = BakedStore::open(probe_tmp.path()).expect("open probe store");
            let mut probe =
                MoeExpertResidencyManager::new(0, MoeExpertResidencyConfig::new(1).unwrap());
            probe
                .register_tensor(
                    &probe_store,
                    0,
                    MoeExpertProjection::GateUp,
                    "model.layers.0.mlp.experts.gate_up_proj",
                    1,
                )
                .expect("register probe tensor");
            let expert_bytes = probe.tensors[0].page_bytes;

            let tmp = synthetic_store(2, expert_bytes);
            let store = BakedStore::open(tmp.path()).expect("open synthetic store");
            let mut manager =
                MoeExpertResidencyManager::new(0, MoeExpertResidencyConfig::new(1).unwrap());
            manager
                .register_tensor(
                    &store,
                    0,
                    MoeExpertProjection::GateUp,
                    "model.layers.0.mlp.experts.gate_up_proj",
                    2,
                )
                .expect("register tensor");

            let e0 = MoeExpertKey {
                layer_idx: 0,
                expert_idx: 0,
                projection: MoeExpertProjection::GateUp,
            };
            let e1 = MoeExpertKey {
                layer_idx: 0,
                expert_idx: 1,
                projection: MoeExpertProjection::GateUp,
            };

            manager.ensure_resident(&store, e0).expect("load expert 0");
            manager
                .prefetch_resident(&store, e1)
                .expect("prefetch expert 1");

            assert!(manager.is_resident(e0));
            assert!(!manager.is_resident(e1));
            assert_eq!(manager.stats().resident_pages, 1);
            assert_eq!(manager.stats().evicted_pages, 0);
            assert_eq!(manager.stats().page_misses, 1);
            assert_eq!(manager.stats().prefetch_requests, 1);
            assert_eq!(manager.stats().prefetch_misses, 1);
            assert_eq!(manager.stats().prefetch_page_misses, 0);
            assert_eq!(manager.stats().prefetch_skipped, 1);
            assert_eq!(manager.stats().prefetch_skipped_pages, 1);
        },
    );
}

#[test]
fn async_prefetch_promotes_before_demand() {
    with_supported_vmm_backend("async_prefetch_promotes_before_demand", |backend| {
        if backend != Backend::Hip {
            return;
        }
        let probe_tmp = synthetic_store(1, 4096);
        let probe_store = BakedStore::open(probe_tmp.path()).expect("open probe store");
        let mut probe =
            MoeExpertResidencyManager::new(0, MoeExpertResidencyConfig::new(1).unwrap());
        probe
            .register_tensor(
                &probe_store,
                0,
                MoeExpertProjection::GateUp,
                "model.layers.0.mlp.experts.gate_up_proj",
                1,
            )
            .expect("register probe tensor");
        let expert_bytes = probe.tensors[0].page_bytes;

        let tmp = synthetic_store(2, expert_bytes);
        let store = BakedStore::open(tmp.path()).expect("open synthetic store");
        let mut manager =
            MoeExpertResidencyManager::new(0, MoeExpertResidencyConfig::new(1).unwrap());
        manager
            .register_tensor(
                &store,
                0,
                MoeExpertProjection::GateUp,
                "model.layers.0.mlp.experts.gate_up_proj",
                2,
            )
            .expect("register tensor");
        manager
            .enable_async_prefetch(1)
            .expect("enable async prefetch");

        let e1 = MoeExpertKey {
            layer_idx: 0,
            expert_idx: 1,
            projection: MoeExpertProjection::GateUp,
        };
        manager
            .prefetch_resident(&store, e1)
            .expect("async prefetch expert 1");
        assert_eq!(manager.stats().async_scheduled_pages, 1);
        manager
            .ensure_resident(&store, e1)
            .expect("demand async expert 1");
        assert!(manager.is_resident(e1));
        assert_eq!(manager.stats().async_completed_pages, 1);
        let promoted = manager
            .resident_pages
            .get(&ResidentPageKey {
                tensor_idx: 0,
                page_offset: expert_bytes,
            })
            .expect("promoted pending page");
        assert_eq!(promoted.last_used, manager.clock);

        let allocation_id = manager
            .resident_weight(0, MoeExpertProjection::GateUp)
            .expect("resident weight")
            .allocation_id()
            .expect("virtual allocation");
        let bytes = manager
            .arena()
            .allocation(allocation_id)
            .expect("allocation")
            .buffer()
            .to_host_range_bytes(expert_bytes, 4096)
            .expect("read expert 1");
        assert!(bytes.iter().all(|b| *b == 2));
    });
}

#[test]
fn page_budget_uses_registered_projection_footprint() {
    with_supported_vmm_backend(
        "page_budget_uses_registered_projection_footprint",
        |_backend| {
            let probe_tmp = synthetic_store(1, 4096);
            let probe_store = BakedStore::open(probe_tmp.path()).expect("open probe store");
            let mut probe =
                MoeExpertResidencyManager::new(0, MoeExpertResidencyConfig::new(1).unwrap());
            probe
                .register_tensor(
                    &probe_store,
                    0,
                    MoeExpertProjection::GateUp,
                    "model.layers.0.mlp.experts.gate_up_proj",
                    1,
                )
                .expect("register probe tensor");
            let page_bytes = probe.tensors[0].page_bytes;

            let expert_bytes = page_bytes + 1;
            let tmp = synthetic_store(2, expert_bytes);
            let store = BakedStore::open(tmp.path()).expect("open synthetic store");
            let mut manager =
                MoeExpertResidencyManager::new(0, MoeExpertResidencyConfig::new(1).unwrap());
            manager
                .register_tensor(
                    &store,
                    0,
                    MoeExpertProjection::GateUp,
                    "model.layers.0.mlp.experts.gate_up_proj",
                    2,
                )
                .expect("register gate/up tensor");
            manager
                .register_tensor(
                    &store,
                    0,
                    MoeExpertProjection::Down,
                    "model.layers.0.mlp.experts.gate_up_proj",
                    2,
                )
                .expect("register down tensor");

            assert_eq!(
                manager
                    .page_budget_for_routed_experts(3)
                    .expect("derive page budget"),
                12
            );
        },
    );
}

#[test]
fn store_range_upload_keeps_stable_virtual_pointer() {
    with_supported_vmm_backend(
        "store_range_upload_keeps_stable_virtual_pointer",
        |_backend| {
            let tmp = synthetic_store(2, 4096);
            let store = BakedStore::open(tmp.path()).expect("open synthetic store");
            let mut arena = VirtualArena::new(0, VirtualBacking::Discard);
            let id = store
                .reserve_virtual_arena(
                    &mut arena,
                    "model.layers.0.mlp.experts.gate_up_proj",
                    VirtualAllocationRole::MoeExpert,
                )
                .expect("reserve tensor");
            let ptr = arena.allocation(id).expect("allocation").buffer().as_ptr();
            store
                .load_range_to_virtual_arena(
                    &mut arena,
                    id,
                    "model.layers.0.mlp.experts.gate_up_proj",
                    4096,
                    4096,
                )
                .expect("upload expert 1");
            let allocation = arena.allocation(id).expect("allocation after upload");
            assert_eq!(allocation.buffer().as_ptr(), ptr);
            let bytes = allocation
                .buffer()
                .to_host_range_bytes(4096, 4096)
                .expect("read expert 1");
            assert!(bytes.iter().all(|b| *b == 2));
        },
    );
}
