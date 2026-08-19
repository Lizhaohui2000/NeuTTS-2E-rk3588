#!/usr/bin/env python3
"""Shared completeness gates for NeuTTS generation sidecars."""

from __future__ import annotations

import json
import re
from pathlib import Path


WORD = re.compile(r"[a-z0-9]+")


def word_count(text: str) -> int:
    return len(WORD.findall(text.lower()))


def classify_generation_status(
    speech_tokens: int,
    predict_limit: int,
    reference_words: int,
    min_speech_tokens_per_word: float = 5.0,
) -> dict[str, object]:
    tokens_per_word = speech_tokens / max(reference_words, 1)
    hit_generation_limit = speech_tokens >= predict_limit
    early_termination = tokens_per_word < min_speech_tokens_per_word
    return {
        "speech_tokens": speech_tokens,
        "predict_limit": predict_limit,
        "speech_tokens_per_word": tokens_per_word,
        "hit_generation_limit": hit_generation_limit,
        "early_termination": early_termination,
        "complete": not hit_generation_limit and not early_termination,
    }


def generation_status(
    audio_path: Path,
    reference_text: str,
    min_speech_tokens_per_word: float = 5.0,
) -> dict[str, object] | None:
    sidecar = audio_path.with_suffix(".json")
    if not sidecar.exists():
        return None
    report = json.loads(sidecar.read_text(encoding="utf-8"))
    return classify_generation_status(
        int(report["per_run"][0]["speech_tokens"]),
        int(report["predict"]),
        word_count(reference_text),
        min_speech_tokens_per_word,
    )


def is_complete(status: dict[str, object] | None) -> bool:
    return status is None or bool(status["complete"])
