from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from reference_strategies import select_reference  # noqa: E402


def test_fixed207_always_uses_safe_prefix():
    for emotion in ("neutral", "happy", "sad", "angry", "surprised"):
        assert select_reference("fixed207", "emily", emotion)["reference_codes"] == 207


def test_routed_strategy_only_promotes_angry():
    assert select_reference("routed103_207", "emily", "happy")["reference_codes"] == 103
    assert select_reference("routed103_207", "emily", "sad")["reference_codes"] == 103
    assert select_reference("routed103_207", "emily", "angry")["reference_codes"] == 207


def test_strategy_rejects_uncalibrated_speaker():
    try:
        select_reference("fixed207", "paul", "happy")
    except ValueError as error:
        assert "calibrated" in str(error)
    else:
        raise AssertionError("uncalibrated speaker was accepted")
