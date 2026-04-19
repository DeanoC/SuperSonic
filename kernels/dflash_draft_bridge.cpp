// Bridge for the DFlash draft's bidirectional attention helper.
//
// Per the per-family codegen-fork rule (memory feedback_hipcc_codegen.md),
// the DFlash helper lives in its own compilation unit. The bridge is a
// thin C-linkage wrapper that the Rust FFI in kernel-ffi/src/dflash.rs
// binds to.

#include "dflash_draft.hip"
