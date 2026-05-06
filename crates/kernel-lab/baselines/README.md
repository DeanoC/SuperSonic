# Kernel Lab Baselines

This directory is the checked-in home for reviewed `kernel-lab` baseline
summaries. Baselines are grouped by GPU architecture:

```text
crates/kernel-lab/baselines/<arch>/<name>.summary.json
crates/kernel-lab/baselines/<arch>/<name>.md
```

Create or refresh a baseline from a reviewed run with:

```bash
cargo run --release -p supersonic-kernel-lab --bin kernel-lab -- baseline \
  --run target/kernel-lab-runs/<run-id> \
  --name required
```

The JSON file is the canonical artifact used by `kernel-lab diff`. The markdown
sidecar is for review and should be regenerated from the same run.
