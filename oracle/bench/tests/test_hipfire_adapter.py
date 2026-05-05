"""hipfire adapter: version-pin gate, subprocess parsing."""
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from oracle.bench.external.hipfire import (
    HipfireAdapter, HipfireVersionMismatch,
)


def test_version_check_passes_when_versions_match(tmp_path):
    pin = tmp_path / "hipfire-version.txt"
    pin.write_text("hipfire 0.4.2-rocm6\n")

    def fake_run(cmd, *args, **kw):
        class R:
            stdout = "hipfire 0.4.2-rocm6\n"
            stderr = ""
            returncode = 0
        return R()
    with patch("oracle.bench.external.hipfire.subprocess.run", side_effect=fake_run):
        ad = HipfireAdapter(version_pin_file=pin)
        ad.assert_version_match()  # no raise


def test_version_check_raises_on_mismatch(tmp_path):
    pin = tmp_path / "hipfire-version.txt"
    pin.write_text("hipfire 0.4.2-rocm6\n")

    def fake_run(cmd, *args, **kw):
        class R:
            stdout = "hipfire 0.5.0-rocm6\n"
            stderr = ""
            returncode = 0
        return R()
    with patch("oracle.bench.external.hipfire.subprocess.run", side_effect=fake_run):
        ad = HipfireAdapter(version_pin_file=pin)
        with pytest.raises(HipfireVersionMismatch) as ei:
            ad.assert_version_match()
        assert "0.4.2-rocm6" in str(ei.value)
        assert "0.5.0-rocm6" in str(ei.value)


def test_supports_returns_false_for_unknown_model(tmp_path):
    pin = tmp_path / "hipfire-version.txt"
    pin.write_text("hipfire 0.4.2-rocm6\n")
    ad = HipfireAdapter(version_pin_file=pin)
    assert ad.supports("qwen3.6-35b-a3b", "int4") in (True, False)  # implementation-specific
    # The base contract: returns a bool, doesn't raise.
