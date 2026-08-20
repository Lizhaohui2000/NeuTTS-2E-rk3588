#!/usr/bin/env python3
"""Build independently encoded short-reference candidates from one waveform.

This is an offline calibration tool. Heavy ASR and codec-encoder dependencies
are intentionally kept out of the RK3588 runtime.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


CODE_RATE_HZ = 50
WORD = re.compile(r"[A-Za-z0-9']+")
INCOMPLETE_START_WORDS = {"a", "an", "and", "for", "is", "of", "or", "that", "to", "us"}


@dataclass(frozen=True)
class Boundary:
    seconds: float
    dbfs: float


@dataclass(frozen=True)
class Window:
    start_seconds: float
    end_seconds: float
    predicted_codes: int
    start_dbfs: float
    end_dbfs: float
    acoustic_score: float


def select_low_energy_boundaries(
    dbfs: list[float],
    hop_seconds: float,
    threshold_dbfs: float,
    min_gap_seconds: float,
) -> list[Boundary]:
    """Select separated local RMS minima below an absolute threshold."""
    if hop_seconds <= 0 or min_gap_seconds < 0:
        raise ValueError("hop must be positive and boundary separation non-negative")
    candidates: list[Boundary] = []
    for index in range(2, len(dbfs) - 2):
        value = float(dbfs[index])
        if value > threshold_dbfs or value != min(dbfs[index - 2 : index + 3]):
            continue
        boundary = Boundary(index * hop_seconds, value)
        if not candidates or boundary.seconds - candidates[-1].seconds >= min_gap_seconds:
            candidates.append(boundary)
        elif boundary.dbfs < candidates[-1].dbfs:
            candidates[-1] = boundary
    return candidates


def enumerate_budget_windows(
    boundaries: list[Boundary],
    min_codes: int,
    max_codes: int,
    max_per_code_count: int = 1,
) -> list[Window]:
    """Enumerate low-energy endpoint pairs inside a speech-code budget."""
    if min_codes <= 0 or max_codes < min_codes:
        raise ValueError("invalid code budget")
    if max_per_code_count <= 0:
        raise ValueError("max_per_code_count must be positive")
    grouped: dict[int, list[Window]] = {}
    for left_index, left in enumerate(boundaries):
        for right in boundaries[left_index + 1 :]:
            duration = right.seconds - left.seconds
            predicted_codes = int(round(duration * CODE_RATE_HZ))
            if predicted_codes < min_codes:
                continue
            if predicted_codes > max_codes:
                break
            # The worse endpoint dominates truncation risk. The mean term
            # distinguishes candidates with the same worst endpoint.
            acoustic_score = max(left.dbfs, right.dbfs) + 0.25 * (
                left.dbfs + right.dbfs
            )
            grouped.setdefault(predicted_codes, []).append(
                Window(
                    start_seconds=left.seconds,
                    end_seconds=right.seconds,
                    predicted_codes=predicted_codes,
                    start_dbfs=left.dbfs,
                    end_dbfs=right.dbfs,
                    acoustic_score=acoustic_score,
                )
            )
    selected = []
    for code_count, windows in grouped.items():
        windows.sort(key=lambda item: (item.acoustic_score, item.start_seconds))
        selected.extend(windows[:max_per_code_count])
    return sorted(selected, key=lambda item: (item.predicted_codes, item.acoustic_score))


def edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for row, expected in enumerate(left, start=1):
        current = [row]
        for column, observed in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected != observed),
                )
            )
        previous = current
    return previous[-1]


def match_source_span(source_text: str, transcript: str) -> dict[str, object]:
    """Match an ASR hypothesis to the closest continuous source-text span."""
    source_matches = list(WORD.finditer(source_text))
    source_words = [match.group().lower() for match in source_matches]
    observed = [match.group().lower() for match in WORD.finditer(transcript)]
    if not source_words or not observed:
        return {
            "reference_text": "",
            "source_match_wer": 1.0,
            "source_span_start_word": None,
            "source_span_end_word": None,
            "source_span_ends_with_punctuation": False,
        }

    best: tuple[float, int, int, int] | None = None
    minimum_width = max(1, len(observed) - 2)
    maximum_width = min(len(source_words), len(observed) + 2)
    for start in range(len(source_words)):
        for width in range(minimum_width, maximum_width + 1):
            end = start + width
            if end > len(source_words):
                break
            errors = edit_distance(source_words[start:end], observed)
            wer = errors / max(width, 1)
            score = (wer, abs(width - len(observed)), start, end)
            if best is None or score < best:
                best = score
    assert best is not None
    wer, _, start, end = best
    start_character = source_matches[start].start()
    end_character = source_matches[end - 1].end()
    while end_character < len(source_text) and source_text[end_character] in ".,?!;:":
        end_character += 1
    reference_text = source_text[start_character:end_character].strip()
    return {
        "reference_text": reference_text,
        "source_match_wer": wer,
        "source_span_start_word": start,
        "source_span_end_word": end,
        "source_span_ends_with_punctuation": reference_text.endswith((".", "?", "!")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--source-text", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-codes", type=int, default=80)
    parser.add_argument("--max-codes", type=int, default=103)
    parser.add_argument("--max-per-code-count", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--boundary-dbfs", type=float, default=-32.0)
    parser.add_argument("--boundary-gap-ms", type=float, default=120.0)
    parser.add_argument("--whisper-model", type=Path, required=True)
    parser.add_argument("--language", default="english")
    parser.add_argument("--codec-model", default="neuphonic/neucodec")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="disallow model downloads; also set HF_HUB_OFFLINE=1 when needed",
    )
    return parser.parse_args()


def transcribe_candidates(
    candidates,
    whisper_model: Path,
    language: str,
    device: str,
    source_text: str,
):
    import torch
    from transformers import pipeline

    pipeline_device = 0 if device.startswith("cuda") else -1
    recognizer = pipeline(
        "automatic-speech-recognition",
        model=str(whisper_model),
        device=pipeline_device,
        model_kwargs={"attn_implementation": "eager"},
    )
    for candidate in candidates:
        result = recognizer(
            {"array": candidate["audio"], "sampling_rate": 16_000},
            return_timestamps="word",
            generate_kwargs={
                "language": language,
                "task": "transcribe",
                "num_beams": 1,
            },
        )
        chunks = result.get("chunks", [])
        text = str(result.get("text", "")).strip()
        final_timestamp = chunks[-1].get("timestamp", (None, None))[1] if chunks else None
        candidate["transcript"] = text
        candidate["word_count"] = len(WORD.findall(text))
        candidate["ends_with_punctuation"] = text.endswith((".", "?", "!"))
        candidate["asr_has_final_timestamp"] = final_timestamp is not None
        source_match = match_source_span(source_text, text)
        candidate.update(source_match)
        first_word = WORD.findall(str(source_match["reference_text"]).lower())
        incomplete_start = bool(first_word and first_word[0] in INCOMPLETE_START_WORDS)
        candidate["incomplete_start_word"] = incomplete_start
        candidate["linguistic_candidate"] = bool(
            candidate["word_count"] >= 4
            and float(source_match["source_match_wer"]) <= 0.25
            and source_match["source_span_ends_with_punctuation"]
            and not incomplete_start
        )
    del recognizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def encode_candidates(candidates, codec_model: str, device: str, local_files_only: bool):
    import torch
    from neucodec import NeuCodec

    codec = NeuCodec.from_pretrained(
        codec_model,
        local_files_only=local_files_only,
    ).eval().to(device)
    for candidate in candidates:
        waveform = torch.from_numpy(candidate["audio"]).float()[None, None].to(device)
        with torch.inference_mode():
            codes = codec.encode_code(waveform).flatten().cpu().tolist()
        candidate["codes"] = [int(code) for code in codes]
        candidate["code_count"] = len(codes)
        candidate["within_budget"] = (
            candidate["min_codes"] <= len(codes) <= candidate["max_codes"]
        )


def main() -> None:
    args = parse_args()
    if args.max_candidates <= 0:
        raise ValueError("max-candidates must be positive")

    import librosa
    import numpy as np
    import soundfile as sf

    audio, sample_rate = librosa.load(args.audio, sr=16_000, mono=True)
    frame_length = 640
    hop_length = 160
    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]
    dbfs = (20.0 * np.log10(np.maximum(rms, 1e-8))).tolist()
    boundaries = select_low_energy_boundaries(
        dbfs,
        hop_length / sample_rate,
        args.boundary_dbfs,
        args.boundary_gap_ms / 1000.0,
    )
    duration_seconds = len(audio) / sample_rate
    endpoint_width = max(1, int(round(0.02 / (hop_length / sample_rate))))
    if float(np.mean(dbfs[:endpoint_width])) <= args.boundary_dbfs:
        boundaries.insert(0, Boundary(0.0, float(np.mean(dbfs[:endpoint_width]))))
    if float(np.mean(dbfs[-endpoint_width:])) <= args.boundary_dbfs:
        boundaries.append(Boundary(duration_seconds, float(np.mean(dbfs[-endpoint_width:]))))

    windows = enumerate_budget_windows(
        boundaries,
        args.min_codes,
        args.max_codes,
        args.max_per_code_count,
    )
    windows.sort(key=lambda item: (item.acoustic_score, item.predicted_codes))
    windows = windows[: args.max_candidates]
    candidates = []
    for window in windows:
        start_sample = max(0, int(round(window.start_seconds * sample_rate)))
        end_sample = min(len(audio), int(round(window.end_seconds * sample_rate)))
        candidate_audio = np.asarray(audio[start_sample:end_sample], dtype=np.float32)
        identifier = (
            f"c{window.predicted_codes:03d}_"
            f"{round(window.start_seconds * 1000):05d}_"
            f"{round(window.end_seconds * 1000):05d}"
        )
        candidates.append(
            {
                "id": identifier,
                **asdict(window),
                "duration_seconds": len(candidate_audio) / sample_rate,
                "audio": candidate_audio,
                "min_codes": args.min_codes,
                "max_codes": args.max_codes,
            }
        )

    transcribe_candidates(
        candidates,
        args.whisper_model,
        args.language,
        args.device,
        args.source_text,
    )
    encode_candidates(candidates, args.codec_model, args.device, args.local_files_only)
    candidates.sort(
        key=lambda item: (
            not item["within_budget"],
            not item["linguistic_candidate"],
            item["code_count"],
            item["acoustic_score"],
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_candidates = []
    speaker_assets = {}
    for rank, candidate in enumerate(candidates, start=1):
        identifier = candidate["id"]
        wav_path = args.output_dir / f"{identifier}.wav"
        code_path = args.output_dir / f"{identifier}.json"
        sf.write(wav_path, candidate["audio"], sample_rate, subtype="PCM_16")
        code_payload = {
            "source": str(args.audio.resolve()),
            "text": candidate["reference_text"],
            "asr_transcript": candidate["transcript"],
            "crop_start_seconds": candidate["start_seconds"],
            "crop_end_seconds": candidate["end_seconds"],
            "codes": candidate["codes"],
        }
        code_path.write_text(json.dumps(code_payload, indent=2) + "\n", encoding="utf-8")
        speaker_assets[identifier] = {
            "text": candidate["reference_text"],
            "codes": candidate["codes"],
        }
        row = {
            key: value
            for key, value in candidate.items()
            if key not in {"audio", "codes", "min_codes", "max_codes"}
        }
        row.update(
            {
                "rank": rank,
                "audio": wav_path.name,
                "code_file": code_path.name,
            }
        )
        manifest_candidates.append(row)

    manifest = {
        "protocol": "neutts-complete-context-budget-search-v1",
        "source_audio": str(args.audio.resolve()),
        "source_text": args.source_text,
        "sample_rate": sample_rate,
        "code_rate_hz": CODE_RATE_HZ,
        "min_codes": args.min_codes,
        "max_codes": args.max_codes,
        "boundary_dbfs": args.boundary_dbfs,
        "boundary_gap_ms": args.boundary_gap_ms,
        "whisper_model": str(args.whisper_model.resolve()),
        "codec_model": args.codec_model,
        "boundaries": [asdict(boundary) for boundary in boundaries],
        "candidates": manifest_candidates,
        "warning": (
            "These are source-level candidates. Release eligibility requires "
            "held-out generation, quality and target-board runtime gates."
        ),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "speakers.json").write_text(
        json.dumps(speaker_assets, indent=2) + "\n",
        encoding="utf-8",
    )
    printable = [
        {
            "rank": item["rank"],
            "id": item["id"],
            "codes": item["code_count"],
            "text": item["transcript"],
            "reference_text": item["reference_text"],
            "source_match_wer": item["source_match_wer"],
            "linguistic_candidate": item["linguistic_candidate"],
            "edge_dbfs": [item["start_dbfs"], item["end_dbfs"]],
        }
        for item in manifest_candidates
    ]
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
