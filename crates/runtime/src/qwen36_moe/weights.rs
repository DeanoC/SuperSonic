//! Runtime-owned Qwen3.6 weight loading and format selection surface.

pub use crate::qwen36_moe::layer_loader::{
    load_to_gpu, resolve_qwen36_store_name, store_contains_qwen36, store_layout_qwen36,
    Qwen36WeightMode, QWEN36_MOE_INT4_GROUP_SIZE, QWEN36_MOE_LOWBIT_GGML_Q4_K,
    QWEN36_MOE_LOWBIT_GGML_Q5_K, QWEN36_MOE_LOWBIT_GGML_Q6_K, QWEN36_MOE_LOWBIT_NATIVE_INT4,
};
