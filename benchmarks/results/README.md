# Published benchmark records

This directory is the source area for validated, publishable benchmark result
records. Keep JSON records small, reviewable, and paired with the versioned
suite manifests and result schema. Model and weight files stay external; a
record identifies them with safe semantic names and SHA-256 digests.

Validate a candidate or committed bundle before publication:

```bash
python3 tools/supersonic-bench.py validate benchmarks/results --publishable
```

Render the disposable static site from the records:

```bash
python3 tools/supersonic-bench.py render benchmarks/results target/benchmarks/site
```

Generated HTML is output, not source, and is intentionally not committed.
The renderer includes only records that pass schema, completeness, quality,
clean-tree, and verified-clock publication checks. Incomplete or diagnostic
runs remain useful locally but do not become pages or aggregate claims.
