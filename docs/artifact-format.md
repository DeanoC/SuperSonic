# Artifact format

The active model source is a custom GQH GGUF pair for `qwen3.8-27b`:

- the primary GGUF contains the model weights and GQH tensors;
- the optional 8192-context GGUF is checked as a separate artifact role;
- the model directory supplies the tokenizer and configuration sidecars;
- the artifact preflight checks existence, readability, and required model
  configuration/tokenizer sidecars before any large test starts;
- Rust startup and artifact tests validate the GQH header and tensor geometry
  after that cheap filesystem preflight.

The canonical runner paths are `/home/deano/gqh-artifacts/` for GGUF files and
`/data/models/Qwen3.8-27B` for the model directory. Repository variables or
secrets may provide equivalent mounted paths, but a missing or mismatched
pair is a hard gate failure.

Use `python3 tools/check-qwen38-artifacts.py --require-8192` for the strict
preflight used by the serial workflow.
