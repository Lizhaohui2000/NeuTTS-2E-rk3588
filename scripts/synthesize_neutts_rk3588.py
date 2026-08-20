#!/usr/bin/env python3
"""Synthesize one NeuTTS-2E utterance with an evaluated reference strategy."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from benchmark_neutts_backbone_server import completion, request_json
from neucodec_rk3588_split_runtime import (
    NPU_CORE_MASK_CHOICES,
    SAMPLE_RATE,
    SplitDecoder,
    write_wav,
)
from neutts_2e_board_runtime import EMOTIONS
from neutts_generation_gates import classify_generation_status, word_count
from reference_strategies import select_reference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--speakers", required=True, type=Path)
    parser.add_argument("--speaker", default="emily")
    parser.add_argument(
        "--strategy",
        choices=("fixed207", "routed103_207", "natural103"),
        default="fixed207",
    )
    parser.add_argument("--emotion", choices=EMOTIONS, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--predict", type=int, default=450)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--frames", type=int, default=450)
    parser.add_argument("--static", action="store_true")
    parser.add_argument(
        "--npu-core-mask",
        choices=NPU_CORE_MASK_CHOICES,
        default="core012",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strategy = select_reference(args.strategy, args.speaker, args.emotion)
    speakers = json.loads(args.speakers.read_text(encoding="utf-8"))
    if args.speaker not in speakers:
        raise ValueError(f"speaker {args.speaker!r} is absent from {args.speakers}")
    stored_codes = speakers[args.speaker]["codes"]
    reference_start = int(strategy["reference_code_start"])
    reference_end = int(strategy["reference_code_end"])
    if len(stored_codes) < reference_end:
        raise ValueError(
            f"speaker reference has {len(stored_codes)} codes, needs index {reference_end}"
        )
    reference_tokens = "".join(
        f"<|speech_{code}|>" for code in stored_codes[reference_start:reference_end]
    )
    prompt = (
        f"<|TEXT_PROMPT_START|>{strategy['reference_text']}"
        f"<|{args.emotion.upper()}|>{args.text}"
        f"<|TEXT_PROMPT_END|><|SPEECH_GENERATION_START|>{reference_tokens}"
    )

    health = request_json(args.url, "/health", args.timeout)
    if health.get("status") != "ok":
        raise RuntimeError(f"llama-server is not ready: {health}")
    decoder = SplitDecoder(
        args.model_dir,
        frames=args.frames,
        dynamic=not args.static,
        npu_core_mask=args.npu_core_mask,
    )
    started = time.perf_counter()
    try:
        backbone = completion(
            args.url,
            prompt,
            args.predict,
            args.temperature,
            args.top_k,
            args.seed,
            args.timeout,
        )
        generation = classify_generation_status(
            int(backbone["speech_tokens"]),
            args.predict,
            word_count(args.text),
        )
        if not generation["complete"]:
            raise RuntimeError(f"generation failed completeness gate: {generation}")
        codec_started = time.perf_counter()
        audio, codec = decoder.decode(backbone["speech_codes"])
        codec_wall_seconds = time.perf_counter() - codec_started
    finally:
        decoder.close()
    wall_seconds = time.perf_counter() - started
    audio_seconds = float(audio.size / SAMPLE_RATE)
    write_wav(args.output, audio)
    report = {
        "model": "NeuTTS-2E RK3588 CPU+NPU",
        **strategy,
        "text": args.text,
        "output": str(args.output),
        "seed": args.seed,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "dynamic_rknn": not args.static,
        "npu_core_mask": args.npu_core_mask,
        "generation": generation,
        "prompt_seconds": float(backbone["prompt_ms"]) / 1000.0,
        "generation_seconds": float(backbone["predicted_ms"]) / 1000.0,
        "codec_wall_seconds": codec_wall_seconds,
        "codec_metrics": codec,
        "audio_seconds": audio_seconds,
        "resident_wall_seconds": wall_seconds,
        "resident_rtf": wall_seconds / audio_seconds,
    }
    metrics_path = args.metrics or args.output.with_suffix(".json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
