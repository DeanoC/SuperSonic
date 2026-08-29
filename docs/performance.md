# Performance

SuperSonic is tuned for maximum measured inference performance on the
Qwen3.8-27B custom GQH GGUF path. Public performance means a reproducible,
correctness-passing record with enough evidence for another contributor to
re-run the same case. No measured performance number is published here yet.

## Headline evidence contract

Do not publish a numeric peer or performance claim unless its direct run
record identifies all of the following:

- engine name and version, repository commit, and ROCm/HIP toolchain;
- verified static GPU provenance, physical-to-logical mapping, architecture,
  and the locked clock/power policy;
- exact artifact identity and digest, including tokenizer and chat-template
  identities when they affect the case, and DFlash2 drafter identity and digest
  for DFlash2 cases;
- exact prompt or stable workload/case, context and generation limits, greedy
  decoding, stop policy, and timing boundary;
- explicit cache state and process state, with `process_reuse=false` for the
  current one-shot evidence;
- raw measured samples; the validator validates their values, suite-required
  count or balanced-round count, and bundle completeness, while the renderer derives sample count, statistic,
  and dispersion from validated raw samples; and
- correctness, including ordinary-versus-MTP token equality and DFlash2
  semantic quality where applicable.

The representative statistic is the median. Raw samples in measured order are
the source of truth. The validator checks their values, required count, and bundle
completeness; the renderer deterministically derives and shows minimum,
maximum, median absolute deviation (MAD), and sample count from those
validated raw samples. Those summaries are not stored source fields. Never
select the best observed sample as the headline value. A quality
failure, missing sample, unexplained token mismatch, incomplete suite, or
unverified clock is a publication blocker.

`uncontrolled-clocks` is a valid diagnostic policy, not a headline policy. Its
recorded telemetry remains useful for troubleshooting, but it is excluded from
headline numbers and peer speedup claims. For locked evidence, isolated GPU
clock transients remain visible but sustained drift means three consecutive loaded
samples outside tolerance; memory clock, power cap, and performance level remain
strict checks. The [benchmark procedure](benchmarks.md) defines the 600-second
quick hard budget and the full 20,700-second minimum within its
21,600-second hard budget, plus the 30/450-minute workflow caps.

## Comparability ruling

Comparability is decided by the validator, not by a page template or by a
reviewer who sees similar names. Two records must match physical hardware,
architecture, clock and power policy, artifact semantics and digest,
DFlash2 drafter semantics and digest for DFlash2 records,
tokenizer/template identity, exact workload and limits, greedy stop behavior,
cache state, warmup policy, process state, measurement boundary, correctness,
and engine/version evidence before a peer ratio is eligible.

Peer artifacts are usually noncomparable by digest under this ruling. A peer
record can remain visible as qualified context with the exact mismatch reasons,
but the site and documents must not calculate or publish a speedup for unlike
artifact digests. The same exclusion applies to unlike cache states, clock
policies, hardware identities, workloads, or timing boundaries.

Prefix-cache cases are explicitly unsupported until the execution adapters
verify empty, populated, and reset transitions. Do not treat a named prefix
state as evidence that a transition happened. The executable suites currently
support only fresh-process `cold-load`; `warm-resident` fails preflight until
warmup and measurement can share one verified resident process.

## Reproduce and review

Run the command in [benchmarks](benchmarks.md), then validate the candidate
before discussing any number:

```bash
python3 tools/supersonic-bench.py validate --publishable <candidate-bundle>
```

For a peer comparison, use the comparator and retain its reasons alongside the
review:

```bash
python3 tools/supersonic-bench.py compare <record-a> <record-b> \
  --output target/benchmarks/candidate/comparison.json
```

Only a complete, quality-passing, schema-valid candidate may be copied into
`benchmarks/results/` by a code-reviewed change. Pages validates those
committed records before rendering and deploys only from the default branch.
With zero committed JSON records it reports the missing baseline and performs
no deployment; the first reviewed baseline is the bootstrap event.
