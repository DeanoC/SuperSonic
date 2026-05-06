from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def sample_windows(ids: list[int], num_samples: int, seqlen: int, seed: int) -> list[list[int]]:
    """Deterministically sample calibration windows from a token stream."""
    if seqlen <= 0:
        raise ValueError("seqlen must be > 0")
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")
    if len(ids) < seqlen + 1:
        raise ValueError(f"not enough calibration tokens ({len(ids)}) for seqlen={seqlen}")

    import random

    rng = random.Random(seed)
    max_start = len(ids) - seqlen - 1
    return [ids[s : s + seqlen] for s in (rng.randrange(0, max_start) for _ in range(num_samples))]


def cache_path(cache_dir: Path, corpus: str, num_samples: int, seqlen: int, seed: int) -> Path:
    safe = corpus.replace("/", "_").replace(":", "_")
    return cache_dir / f"{safe}-n{num_samples}-t{seqlen}-seed{seed}.json"


def load_or_build_wikitext2_calibration(
    *,
    tokenizer: Any,
    cache_dir: Path,
    num_samples: int,
    seqlen: int,
    seed: int,
    log: Any = print,
) -> list[list[int]]:
    """Load or build deterministic WikiText-2 calibration token windows.

    Stored as JSON rather than torch-specific pickle so cache files remain
    inspectable and portable across Python/Torch versions.
    """
    corpus = "wikitext-2-raw-v1_train"
    path = cache_path(cache_dir, corpus, num_samples, seqlen, seed)
    if path.is_file():
        payload = json.loads(path.read_text())
        if (
            payload.get("schema") == 1
            and payload.get("num_samples") == num_samples
            and payload.get("seqlen") == seqlen
            and payload.get("seed") == seed
            and isinstance(payload.get("windows"), list)
        ):
            log(f"[calib] loaded token cache {path}")
            return payload["windows"]

    log("[calib] loading WikiText-2 train split via `datasets`...")
    from datasets import load_dataset

    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(r["text"] for r in train if r["text"].strip())
    enc = tokenizer(text, return_tensors=None)
    ids = list(enc["input_ids"])
    log(f"[calib] tokenized train: {len(ids)} tokens")
    windows = sample_windows(ids, num_samples, seqlen, seed)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {
                "schema": 1,
                "corpus": corpus,
                "num_samples": num_samples,
                "seqlen": seqlen,
                "seed": seed,
                "token_count": len(ids),
                "windows": windows,
            }
        )
    )
    tmp.replace(path)
    log(f"[calib] wrote token cache {path}")
    return windows
