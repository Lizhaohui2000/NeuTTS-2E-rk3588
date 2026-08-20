from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from reference_strategies import (  # noqa: E402
    STRATEGY_CONFIG,
    load_reference_codes,
    load_strategy_config,
    select_reference,
)


CONFIG = load_strategy_config()
EMOTIONS = ("neutral", "happy", "sad", "angry", "surprised")


def test_stable_always_uses_safe_prefix():
    for emotion in EMOTIONS:
        reference = select_reference("stable", "emily", emotion, CONFIG)
        assert reference["reference_code_start"] == 0
        assert reference["reference_code_end"] == 207
        assert reference["fallback_used"] is False


def test_fast_falls_back_only_for_unvalidated_emotion():
    for emotion in ("neutral", "happy", "sad", "surprised"):
        reference = select_reference("fast", "emily", emotion, CONFIG)
        assert reference["reference_id"] == "prefix103"
        assert reference["fallback_used"] is False
    angry = select_reference("fast", "emily", "angry", CONFIG)
    assert angry["reference_id"] == "prefix207"
    assert angry["fallback_used"] is True
    assert "not release-validated" in angry["fallback_reason"]

    for emotion in ("fearful", "disgusted"):
        reference = select_reference("fast", "emily", emotion, CONFIG)
        assert reference["reference_id"] == "prefix207"
        assert reference["fallback_used"] is True


def test_legacy_strategy_names_remain_compatible():
    assert select_reference("fixed207", "emily", "happy", CONFIG)["strategy"] == "stable"
    assert select_reference("routed103_207", "emily", "happy", CONFIG)["strategy"] == "fast"
    assert (
        select_reference("natural103", "emily", "angry", CONFIG)["strategy"]
        == "experimental_natural103_slice"
    )


def test_experimental_slice_is_not_presented_as_release_mode():
    reference = select_reference(
        "experimental_natural103_slice", "emily", "angry", CONFIG
    )
    assert reference["reference_code_start"] == 103
    assert reference["reference_code_end"] == 206
    assert reference["release_status"] == "experimental"


def test_independently_encoded_control_loads_103_codes():
    reference = select_reference(
        "experimental_reencoded103", "emily", "angry", CONFIG
    )
    codes = load_reference_codes(
        reference,
        {"emily": {"codes": [0] * 207}},
        STRATEGY_CONFIG.parent,
    )
    assert reference["reference_source"] == "code_file"
    assert len(codes) == 103
    assert codes[:3] == [50307, 5778, 14692]


def test_speaker_code_source_is_range_checked():
    reference = select_reference("stable", "emily", "happy", CONFIG)
    codes = load_reference_codes(
        reference,
        {"EMILY": {"codes": list(range(207))}},
        STRATEGY_CONFIG.parent,
    )
    assert codes == list(range(207))


def test_strategy_rejects_uncalibrated_speaker():
    try:
        select_reference("stable", "paul", "happy", CONFIG)
    except ValueError as error:
        assert "no calibrated reference profile" in str(error)
    else:
        raise AssertionError("uncalibrated speaker was accepted")


def test_release_mode_fails_closed_without_valid_fallback():
    config = {
        "schema_version": 2,
        "aliases": {},
        "speakers": {
            "test": {
                "references": {
                    "short": {
                        "source": "speaker_codes",
                        "start": 0,
                        "count": 2,
                        "text": "complete phrase.",
                        "release_eligible": False,
                        "validated_emotions": [],
                    }
                },
                "modes": {
                    "fast": {
                        "default_reference": "short",
                        "fallback_reference": "missing",
                        "require_release_eligible": True,
                    }
                },
            }
        },
    }
    try:
        select_reference("fast", "test", "happy", config)
    except ValueError as error:
        assert "fallback_reference" in str(error)
    else:
        raise AssertionError("release mode accepted an invalid fallback")
