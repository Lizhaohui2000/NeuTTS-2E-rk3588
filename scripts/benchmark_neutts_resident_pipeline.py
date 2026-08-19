#!/usr/bin/env python3
"""Measure a warm resident llama-server -> RKNN NeuCodec serial pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from benchmark_neutts_backbone_server import completion, request_json, summarize
from neucodec_rk3588_split_runtime import (
    NPU_CORE_MASK_CHOICES,
    SAMPLE_RATE,
    SplitDecoder,
    write_wav,
)
from neutts_2e_board_runtime import EMOTIONS


SYSFS = {
    "temperature_millic": Path("/sys/class/thermal/thermal_zone0/temp"),
    "cpu_policy4_hz": Path("/sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq"),
    "cpu_policy6_hz": Path("/sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq"),
    "dmc_hz": Path("/sys/class/devfreq/dmc/cur_freq"),
    "npu_hz": Path("/sys/class/devfreq/fdab0000.npu/cur_freq"),
}


def hardware_snapshot() -> dict[str, int | None]:
    values: dict[str, int | None] = {}
    for name, path in SYSFS.items():
        try:
            values[name] = int(path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            values[name] = None
    return values


def code_hash(codes: list[int]) -> str:
    raw = ",".join(str(code) for code in codes).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--speakers", type=Path, required=True)
    parser.add_argument("--speaker", default="emily")
    parser.add_argument(
        "--reference-code-limit",
        type=int,
        help="use only this many leading reference speech codes",
    )
    parser.add_argument(
        "--reference-text",
        help="transcript matching a truncated reference-code prefix",
    )
    parser.add_argument("--emotion", choices=EMOTIONS, default="sad")
    parser.add_argument(
        "--text",
        default="This is a steady state board benchmark for expressive speech synthesis.",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=450)
    parser.add_argument("--predict", type=int, default=450)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--cache-prompt",
        action="store_true",
        help="reuse an identical complete prompt; unsuitable for changing synthesis text",
    )
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument(
        "--npu-core-mask",
        choices=NPU_CORE_MASK_CHOICES,
        default="core012",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metrics", type=Path)
    args = parser.parse_args()

    speakers = json.loads(args.speakers.read_text(encoding="utf-8"))
    reference = speakers[args.speaker]
    reference_codes = reference["codes"]
    if args.reference_code_limit is not None:
        if not 0 < args.reference_code_limit <= len(reference_codes):
            raise ValueError("reference-code-limit must be within the stored reference")
        if not args.reference_text:
            raise ValueError("reference-text is required when truncating reference codes")
        reference_codes = reference_codes[: args.reference_code_limit]
    reference_text = args.reference_text or reference["text"]
    reference_tokens = "".join(f"<|speech_{code}|>" for code in reference_codes)
    prompt = (
        f"<|TEXT_PROMPT_START|>{reference_text}<|{args.emotion.upper()}|>{args.text}"
        f"<|TEXT_PROMPT_END|><|SPEECH_GENERATION_START|>{reference_tokens}"
    )
    health = request_json(args.url, "/health", args.timeout)
    props = request_json(args.url, "/props", args.timeout)
    decoder = SplitDecoder(
        args.model_dir,
        frames=args.frames,
        dynamic=args.dynamic,
        npu_core_mask=args.npu_core_mask,
    )

    def run_once() -> tuple[dict[str, object], np.ndarray]:
        before = hardware_snapshot()
        started = time.perf_counter()
        backbone = completion(
            args.url,
            prompt,
            args.predict,
            args.temperature,
            args.top_k,
            args.seed,
            args.timeout,
            cache_prompt=args.cache_prompt,
        )
        codec_started = time.perf_counter()
        audio, codec = decoder.decode(backbone["speech_codes"])
        codec_wall_seconds = time.perf_counter() - codec_started
        pipeline_wall_seconds = time.perf_counter() - started
        audio_seconds = float(audio.size / SAMPLE_RATE)
        predicted_seconds = float(backbone["predicted_ms"]) / 1000.0
        row = {
            "speech_tokens": int(backbone["speech_tokens"]),
            "speech_code_sha256": str(backbone["speech_code_sha256"]),
            "audio_seconds": audio_seconds,
            "backbone_request_seconds": float(backbone["wall_seconds"]),
            "backbone_prompt_tokens": int(backbone["prompt_n"]),
            "backbone_prompt_seconds": float(backbone["prompt_ms"]) / 1000.0,
            "backbone_prompt_tokens_per_second": float(backbone["prompt_per_second"]),
            "backbone_generation_seconds": predicted_seconds,
            "backbone_generation_rtf": float(backbone["steady_backbone_rtf"]),
            "codec_wall_seconds": codec_wall_seconds,
            "codec_wall_rtf": codec_wall_seconds / audio_seconds,
            "pipeline_wall_seconds": pipeline_wall_seconds,
            "measured_resident_rtf": pipeline_wall_seconds / audio_seconds,
            "generation_plus_codec_rtf": (predicted_seconds + codec_wall_seconds)
            / audio_seconds,
            "hardware_before": before,
            "hardware_after": hardware_snapshot(),
        }
        return row, audio

    last_audio: np.ndarray | None = None
    rows: list[dict[str, object]] = []
    try:
        for _ in range(args.warmup_runs):
            _, last_audio = run_once()
        for _ in range(args.runs):
            row, last_audio = run_once()
            rows.append(row)
    finally:
        decoder.close()

    code_hashes = [str(row["speech_code_sha256"]) for row in rows]
    fields = (
        "backbone_generation_rtf",
        "codec_wall_rtf",
        "generation_plus_codec_rtf",
        "measured_resident_rtf",
    )
    report = {
        "mode": "measured warm resident NeuTTS serial pipeline",
        "server_url": args.url,
        "server_health": health,
        "model_alias": props.get("model_alias"),
        "speaker": args.speaker,
        "reference_codes": len(reference_codes),
        "reference_text": reference_text,
        "emotion": args.emotion,
        "text": args.text,
        "predict": args.predict,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "seed": args.seed,
        "cache_prompt": args.cache_prompt,
        "dynamic": args.dynamic,
        "npu_core_mask": args.npu_core_mask,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "warmup_runs": args.warmup_runs,
        "runs": args.runs,
        "all_token_sequences_identical": len(set(code_hashes)) == 1,
        "token_sequence_sha256": code_hashes[0],
        "metrics": {
            field: summarize([float(row[field]) for row in rows]) for field in fields
        },
        "per_run": rows,
        "interpretation": (
            "measured_resident_rtf includes llama-server request/prompt overhead and serial "
            "codec wall time, but excludes process/model initialization. "
            "generation_plus_codec_rtf uses llama.cpp generation timing plus codec wall time."
        ),
    }
    if last_audio is not None:
        report["audio_float32_sha256"] = hashlib.sha256(
            np.asarray(last_audio, dtype="<f4").tobytes()
        ).hexdigest()
        if args.output:
            write_wav(args.output, last_audio)
            report["output"] = str(args.output)
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
