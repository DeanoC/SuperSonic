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

CI compares pull-request candidates against both the PR base ref and the checked-in
`gfx1100/required.summary.json` baseline. Manual `kernel-lab` workflow runs expose
presets:

- `pr`: required suite with short timing, matching pull-request CI. Checked-baseline
  comparison is report-only; PR gating comes from the base-ref comparison.
- `required`: required suite with the default local timing depth.
- `everything`: all registered tasks, including optional stress tasks.
- `stress`: only tasks tagged `stress`; checked-baseline comparison is off by default.

When `tasks` is manually overridden, checked-baseline comparison is disabled in
`auto` mode because subset runs do not contain every task from the required
baseline. Set `checked_baseline=true` only for overrides that still include the
required baseline tasks.
