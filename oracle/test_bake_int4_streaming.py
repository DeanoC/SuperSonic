#!/usr/bin/env python3
"""
Unit tests for `StreamingPackageWriter` in `bake_int4.py`.

The producer side of the bake (GPTQ driver + fused-expert pass) was
accumulating every quantized tensor's bytes in a host-RAM list until a
final flush, which OOM'd on 35B-A3B near the end of a run. The writer
class replaces that with a streaming sink that opens `weights.bin` once,
writes tensor bytes as they arrive, and emits `manifest.json` on
`finalize`. These tests pin the on-disk format invariants the Rust loader
(`crates/model-store/src/store.rs`) relies on:

- 4096-byte alignment between successive tensors (`align_up` cursor)
- byte-exact round-trip for every written tensor
- duplicate tensor names rejected (catches a bake bug, not a runtime bug)
- manifest entries sorted alphabetically (stable diffs across runs)
- error path leaves no `manifest.json` (consumers can detect partial bakes)
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bake_int4 import (
    CONVERTER_VERSION,
    FORMAT_VERSION,
    StreamingPackageWriter,
    align_up,
)


class StreamingPackageWriterTest(unittest.TestCase):
    def _open(self, td: str) -> StreamingPackageWriter:
        return StreamingPackageWriter(Path(td) / "bake", model_family="qwen35")

    def test_alignment_and_offsets(self) -> None:
        """Each tensor's byte offset is 4096-aligned, and successive tensors
        are placed back-to-back with zero-padding between."""
        with TemporaryDirectory() as td:
            with self._open(td) as w:
                w.write_tensor("a", b"\x01" * 100, [100], "u8", "Raw")
                w.write_tensor("b", b"\x02" * 200, [200], "u8", "Raw")
                w.write_tensor("c", b"\x03" * 5000, [5000], "u8", "Raw")
            manifest = json.loads((Path(td) / "bake" / "manifest.json").read_text())
            entries = {e["name"]: e for e in manifest["tensors"]}
            self.assertEqual(entries["a"]["offset"], 0)
            self.assertEqual(entries["a"]["byte_len"], 100)
            self.assertEqual(entries["b"]["offset"], 4096)
            self.assertEqual(entries["b"]["byte_len"], 200)
            # 4096 + 200 = 4296 → rounded up to 8192.
            self.assertEqual(entries["c"]["offset"], 8192)
            self.assertEqual(entries["c"]["byte_len"], 5000)
            # File length matches the last tensor's end.
            file_len = (Path(td) / "bake" / "weights.bin").stat().st_size
            self.assertEqual(file_len, 8192 + 5000)

    def test_byte_exact_roundtrip(self) -> None:
        """Reading bytes back from `weights.bin` at the manifest-recorded
        offset yields exactly what was written. Catches the kind of
        zero-padding-of-data-bytes bug that's invisible until the runtime
        tries to use the tensor."""
        payloads = {
            "embed": bytes(range(256)) * 4,
            "scale": b"\xab\xcd" * 100,
            "tile":  b"\x12" * 1024,
        }
        with TemporaryDirectory() as td:
            with self._open(td) as w:
                for name, data in payloads.items():
                    w.write_tensor(name, data, [len(data)], "u8", "Raw")
            manifest = json.loads((Path(td) / "bake" / "manifest.json").read_text())
            entries = {e["name"]: e for e in manifest["tensors"]}
            wb = (Path(td) / "bake" / "weights.bin").read_bytes()
            for name, data in payloads.items():
                e = entries[name]
                start = e["offset"]
                end = start + e["byte_len"]
                self.assertEqual(wb[start:end], data,
                                 f"byte-roundtrip mismatch for {name!r}")

    def test_duplicate_name_raises(self) -> None:
        """Writing the same tensor name twice is a bake bug — silently
        ignoring it would leave one of the two payloads orphaned in
        `weights.bin` while the manifest pointed at the other."""
        with TemporaryDirectory() as td:
            w = StreamingPackageWriter(Path(td) / "bake", model_family="qwen35")
            try:
                w.write_tensor("dup", b"x" * 10, [10], "u8", "Raw")
                with self.assertRaises(ValueError):
                    w.write_tensor("dup", b"y" * 10, [10], "u8", "Raw")
            finally:
                # Don't call finalize — leaves no manifest, exactly the
                # behaviour we want on error.
                w._fh.close()  # noqa: SLF001 — test cleanup

    def test_has_reflects_writes(self) -> None:
        with TemporaryDirectory() as td:
            with self._open(td) as w:
                self.assertFalse(w.has("a"))
                w.write_tensor("a", b"x" * 4, [4], "u8", "Raw")
                self.assertTrue(w.has("a"))
                self.assertFalse(w.has("b"))

    def test_manifest_sorted_alphabetically(self) -> None:
        """Manifest entries land alphabetically regardless of write order.
        On-disk order in `weights.bin` follows write order (the runtime
        keys by name into a HashMap so on-disk order is irrelevant); the
        manifest sort is purely for stable diffs."""
        with TemporaryDirectory() as td:
            with self._open(td) as w:
                for name in ["zeta", "alpha", "mu", "beta"]:
                    w.write_tensor(name, b"x" * 8, [8], "u8", "Raw")
            manifest = json.loads((Path(td) / "bake" / "manifest.json").read_text())
            names = [e["name"] for e in manifest["tensors"]]
            self.assertEqual(names, sorted(names))
            self.assertEqual(names, ["alpha", "beta", "mu", "zeta"])

    def test_manifest_header(self) -> None:
        with TemporaryDirectory() as td:
            with self._open(td) as w:
                w.write_tensor("a", b"x", [1], "u8", "Raw")
            manifest = json.loads((Path(td) / "bake" / "manifest.json").read_text())
            self.assertEqual(manifest["format_version"], FORMAT_VERSION)
            self.assertEqual(manifest["converter_version"], CONVERTER_VERSION)
            self.assertEqual(manifest["model_family"], "qwen35")

    def test_partial_bake_skips_manifest(self) -> None:
        """If the producer raises mid-write, the manifest must not be
        emitted — otherwise a downstream consumer might mmap weights.bin
        and read truncated/missing tensors as if they were complete."""
        bake = None
        with TemporaryDirectory() as td:
            bake = Path(td) / "bake"
            try:
                with StreamingPackageWriter(bake, model_family="qwen35") as w:
                    w.write_tensor("a", b"x" * 16, [16], "u8", "Raw")
                    raise RuntimeError("simulated GPTQ failure")
            except RuntimeError:
                pass
            self.assertTrue((bake / "weights.bin").exists(),
                            "weights.bin should be created even on partial run")
            self.assertFalse((bake / "manifest.json").exists(),
                             "manifest.json must NOT be written on partial run")

    def test_align_up_helper(self) -> None:
        # Sanity-check the helper the writer relies on; pads to next 4096.
        self.assertEqual(align_up(0, 4096), 0)
        self.assertEqual(align_up(1, 4096), 4096)
        self.assertEqual(align_up(4095, 4096), 4096)
        self.assertEqual(align_up(4096, 4096), 4096)
        self.assertEqual(align_up(4097, 4096), 8192)


if __name__ == "__main__":
    unittest.main()
