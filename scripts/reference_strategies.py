#!/usr/bin/env python3
"""Reference-prefix selection for the evaluated NeuTTS-2E strategies."""

from __future__ import annotations

import json
from pathlib import Path


STRATEGY_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "reference_strategies.json"


def load_strategy_config(path: Path = STRATEGY_CONFIG) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_reference(
    strategy: str,
    speaker: str,
    emotion: str,
    config: dict[str, object] | None = None,
) -> dict[str, object]:
    config = config or load_strategy_config()
    expected_speaker = str(config["speaker"])
    if speaker.lower() != expected_speaker:
        raise ValueError(
            f"strategy prefixes are calibrated for {expected_speaker!r}, got {speaker!r}"
        )
    strategies = config["strategies"]
    if strategy not in strategies:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {sorted(strategies)}")
    profile = strategies[strategy]
    code_count = int(
        profile["emotion_overrides"].get(
            emotion.lower(), profile["default_reference_codes"]
        )
    )
    prefix_text = config["prefixes"].get(str(code_count))
    if prefix_text is None:
        raise ValueError(f"strategy selected an undefined {code_count}-token prefix")
    return {
        "strategy": strategy,
        "speaker": expected_speaker,
        "emotion": emotion.lower(),
        "reference_codes": code_count,
        "reference_text": str(prefix_text),
    }
