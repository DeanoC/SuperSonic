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
  identities when they affect the case;
- exact prompt or stable workload/case, context and generation limits, greedy
  decoding, stop policy, and timing boundary;
- explicit cache state and process state, with `process_reuse=false` for the
  current one-shot evidence;
- raw measured samples, sample count, statistic, and dispersion; and
- correctness, including ordinary-versus-MTP token equality where applicable.

The representative statistic is the median. Keep raw samples in measured
order and show minimum, maximum, median absolute deviation (MAD), and sample
count. Never select the best observed sample as the headline value. A quality
failure, missing sample, unexplained token mismatch, incomplete suite, or
unverified clock is a publication blocker.

`uncontrolled-clocks` is a valid diagnostic policy, not a headline policy. Its
recorded telemetry remains useful for troubleshooting, but it is excluded from
headline numbers and peer speedup claims. The [benchmark procedure](benchmarks.md)
defines the exact 600-second quick and 21,600-second full budgets and the
30/390-minute workflow caps.

## Comparability ruling

Comparability is decided by the validator, not by a page template or by a
reviewer who sees similar names. Two records must match physical hardware,
architecture, clock and power policy, artifact semantics and digest,
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
state as evidence that a transition happened. For the supported `cold-load`
and `warm-resident` series, the current records retain `process_reuse=false`.

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
