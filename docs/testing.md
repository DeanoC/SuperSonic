# Testing

The active correctness gate covers the Qwen3.8-27B GQH artifact on HIP. The
large artifact is resident during the crawl, so every ignored gate is run
serially with one Rust test thread.

## CPU contract checks

The pull-request gate compiles for `gfx1201` without requiring a GPU:

```bash
cargo fmt --all --check
HIP_ARCH=gfx1201 cargo check --workspace --all-targets
python3 tools/check-support-matrix.py
python3 tools/check-active-docs.py
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

## Artifact correctness gate

On the R9700 runner, preflight the artifact pair, then use the release build
for the focused kernel tests, complete crawl, decode/chat tests, and ordinary
versus MTP token comparison:

```bash
RUST_TEST_THREADS=1 cargo test --release -p kernel-ffi --lib 'gqh::tests::' \
  -- --include-ignored --test-threads=1
RUST_TEST_THREADS=1 cargo test --release -p qwen38 --test qwen38_gqh_gguf_crawl \
  -- --include-ignored --test-threads=1
RUST_TEST_THREADS=1 cargo test --release -p supersonic-runtime \
  --test qwen38_gqh_decode_rung11 -- --include-ignored --test-threads=1
```

Do not remove `--include-ignored` or `--test-threads=1`: the crawl is an
explicit correctness gate and parallel execution can exhaust device memory.
