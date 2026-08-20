#!/usr/bin/env python3
"""Reference-profile selection for NeuTTS-2E release and research modes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


STRATEGY_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "reference_strategies.json"
VALID_CODE_SOURCES = {"speaker_codes", "code_file"}


def load_strategy_config(path: Path = STRATEGY_CONFIG) -> dict[str, object]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 2:
        raise ValueError(f"unsupported reference config schema in {path}")
    if not isinstance(config.get("speakers"), dict):
        raise ValueError(f"reference config has no speakers mapping: {path}")
    return config


def _speaker_profile(config: Mapping[str, object], speaker: str) -> tuple[str, Mapping[str, object]]:
    speakers = config["speakers"]
    assert isinstance(speakers, Mapping)
    speaker_key = next((name for name in speakers if str(name).lower() == speaker.lower()), None)
    if speaker_key is None:
        available = ", ".join(sorted(str(name) for name in speakers)) or "none"
        raise ValueError(
            f"speaker {speaker!r} has no calibrated reference profile; available: {available}"
        )
    profile = speakers[speaker_key]
    if not isinstance(profile, Mapping):
        raise ValueError(f"speaker profile {speaker_key!r} must be an object")
    return str(speaker_key), profile


def _emotion_is_validated(reference: Mapping[str, object], emotion: str) -> bool:
    validated = reference.get("validated_emotions", [])
    return isinstance(validated, list) and emotion.lower() in {
        str(item).lower() for item in validated
    }


def _validate_reference(reference_id: str, reference: Mapping[str, object]) -> None:
    source = str(reference.get("source", ""))
    if source not in VALID_CODE_SOURCES:
        raise ValueError(
            f"reference {reference_id!r} has unsupported source {source!r}; "
            f"choose from {sorted(VALID_CODE_SOURCES)}"
        )
    start = reference.get("start")
    count = reference.get("count")
    if not isinstance(start, int) or start < 0:
        raise ValueError(f"reference {reference_id!r} has an invalid start")
    if not isinstance(count, int) or count <= 0:
        raise ValueError(f"reference {reference_id!r} has an invalid count")
    if not str(reference.get("text", "")).strip():
        raise ValueError(f"reference {reference_id!r} has no matching transcript")
    if source == "code_file" and not str(reference.get("path", "")).strip():
        raise ValueError(f"reference {reference_id!r} has no code-file path")


def select_reference(
    strategy: str,
    speaker: str,
    emotion: str,
    config: dict[str, object] | None = None,
) -> dict[str, object]:
    """Select a reference and apply the release-mode validation fallback."""
    config = config or load_strategy_config()
    speaker_key, speaker_profile = _speaker_profile(config, speaker)
    aliases = config.get("aliases", {})
    if not isinstance(aliases, Mapping):
        raise ValueError("reference config aliases must be an object")
    canonical_strategy = str(aliases.get(strategy, strategy))
    modes = speaker_profile.get("modes")
    references = speaker_profile.get("references")
    if not isinstance(modes, Mapping) or not isinstance(references, Mapping):
        raise ValueError(f"speaker profile {speaker_key!r} needs modes and references")
    if canonical_strategy not in modes:
        available = sorted(str(name) for name in modes)
        raise ValueError(f"unknown strategy {strategy!r}; choose from {available}")
    mode = modes[canonical_strategy]
    if not isinstance(mode, Mapping):
        raise ValueError(f"strategy {canonical_strategy!r} must be an object")

    overrides = mode.get("emotion_overrides", {})
    if not isinstance(overrides, Mapping):
        raise ValueError(f"strategy {canonical_strategy!r} emotion_overrides must be an object")
    emotion_key = emotion.lower()
    reference_id = str(overrides.get(emotion_key, mode.get("default_reference", "")))
    if reference_id not in references:
        raise ValueError(
            f"strategy {canonical_strategy!r} selected undefined reference {reference_id!r}"
        )
    reference = references[reference_id]
    if not isinstance(reference, Mapping):
        raise ValueError(f"reference {reference_id!r} must be an object")

    fallback_used = False
    fallback_reason = None
    if mode.get("require_release_eligible") and (
        not bool(reference.get("release_eligible"))
        or not _emotion_is_validated(reference, emotion_key)
    ):
        fallback_id = str(mode.get("fallback_reference", ""))
        fallback = references.get(fallback_id)
        if not isinstance(fallback, Mapping):
            raise ValueError(
                f"strategy {canonical_strategy!r} needs valid fallback_reference"
            )
        if not bool(fallback.get("release_eligible")):
            raise ValueError(
                f"fallback reference {fallback_id!r} is not release-eligible"
            )
        fallback_used = True
        fallback_reason = f"{reference_id} is not release-validated for {emotion_key}"
        reference_id = fallback_id
        reference = fallback

    _validate_reference(reference_id, reference)
    start = int(reference["start"])
    count = int(reference["count"])
    return {
        "requested_strategy": strategy,
        "strategy": canonical_strategy,
        "release_status": str(mode.get("release_status", "experimental")),
        "speaker": speaker_key,
        "emotion": emotion_key,
        "reference_id": reference_id,
        "reference_source": str(reference["source"]),
        "reference_code_file": reference.get("path"),
        "reference_code_start": start,
        "reference_code_end": start + count,
        "reference_codes": count,
        "reference_text": str(reference["text"]),
        "reference_validation_status": str(
            reference.get("validation_status", "unvalidated")
        ),
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
    }


def load_reference_codes(
    selection: Mapping[str, object],
    speakers: Mapping[str, object],
    config_dir: Path,
) -> list[int]:
    """Load selected codes from the speaker asset or a standalone JSON file."""
    source = str(selection["reference_source"])
    if source == "speaker_codes":
        speaker = str(selection["speaker"])
        speaker_asset_key = next(
            (name for name in speakers if str(name).lower() == speaker.lower()), None
        )
        asset = speakers.get(speaker_asset_key) if speaker_asset_key is not None else None
        if not isinstance(asset, Mapping) or not isinstance(asset.get("codes"), list):
            raise ValueError(f"speaker {speaker!r} is absent from the speaker asset")
        available_codes = asset["codes"]
    elif source == "code_file":
        raw_path = Path(str(selection["reference_code_file"]))
        code_path = raw_path if raw_path.is_absolute() else config_dir / raw_path
        payload = json.loads(code_path.read_text(encoding="utf-8"))
        available_codes = payload.get("codes") if isinstance(payload, Mapping) else payload
        if not isinstance(available_codes, list):
            raise ValueError(f"reference code file has no codes list: {code_path}")
    else:
        raise ValueError(f"unsupported reference source {source!r}")

    start = int(selection["reference_code_start"])
    end = int(selection["reference_code_end"])
    if len(available_codes) < end:
        raise ValueError(
            f"reference source has {len(available_codes)} codes, needs index {end}"
        )
    codes = available_codes[start:end]
    if any(
        isinstance(code, bool)
        or not isinstance(code, int)
        or code < 0
        or code >= 4**8
        for code in codes
    ):
        raise ValueError("reference codes must be integers in [0, 65535]")
    return codes
