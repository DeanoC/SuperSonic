# SuperSonic Kernel Lab

- run: `2026-05-06-e25cd03`
- git: `e25cd03`
- backend: `HIP` device `0` arch `gfx1100`
- required: 5/5

| task | correct | timing | median us |
| --- | --- | --- | ---: |
| `qwen35.full_attention_prefill` | yes | `hip_event` | 91.520 |
| `qwen36.batched_prefill_attn_full` | yes | `hip_event` | 127.001 |
| `qwen36.router_permute` | yes | `hip_event` | 35.680 |
| `qwen36.grouped_expert_int4` | yes | `hip_event` | 249.502 |
| `qwen36.unpermute_combine` | yes | `hip_event` | 36.780 |
