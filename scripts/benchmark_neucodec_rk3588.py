#!/usr/bin/env python3
"""Benchmark a resident RK3588 NeuCodec decoder without model-load time."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from neucodec_rk3588_split_runtime import (
    NPU_CORE_MASK_CHOICES,
    SplitDecoder,
    write_wav,
)


SPEECH_TOKEN = re.compile(rb"<\|speech_(\d+)\|>")


def load_codes(path: Path) -> list[int]:
    raw = path.read_bytes()
    marker = raw.rfind(b"<|SPEECH_GENERATION_START|>")
    if marker < 0:
        raise ValueError(f"{path} has no speech generation marker")
    generation = raw[marker:]
    truncation_marker = b"... (truncated)\n"
    if truncation_marker in generation:
        generation = generation.split(truncation_marker, 1)[1]
    generation = generation.split(b"<|SPEECH_GENERATION_END|>", 1)[0]
    codes = [int(match.group(1)) for match in SPEECH_TOKEN.finditer(generation)]
    if not codes:
        raise ValueError(f"{path} has no generated speech tokens")
    return codes


def load_codes_from_backbone_report(path: Path) -> list[int]:
    report = json.loads(path.read_text(encoding="utf-8"))
    runs = report.get("per_run")
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"{path} has no backbone benchmark runs")
    sequences = [row.get("speech_codes") for row in runs]
    if any(not isinstance(codes, list) or not codes for codes in sequences):
        raise ValueError(f"{path} has a run without speech codes")
    if any(codes != sequences[0] for codes in sequences[1:]):
        raise ValueError(f"{path} backbone runs do not use one identical token sequence")
    return [int(code) for code in sequences[0]]


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    code_source = parser.add_mutually_exclusive_group(required=True)
    code_source.add_argument("--codes-log", type=Path)
    code_source.add_argument(
        "--backbone-report",
        type=Path,
        help="resident llama-server report containing per-run speech_codes",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--frames", type=int, default=450)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument(
        "--npu-core-mask",
        choices=NPU_CORE_MASK_CHOICES,
        default="auto",
    )
    parser.add_argument(
        "--transformer-00-model",
        help="optional model filename replacing Transformer block 0 for isolated A/B tests",
    )
    args = parser.parse_args()

    codes = (
        load_codes_from_backbone_report(args.backbone_report)
        if args.backbone_report
        else load_codes(args.codes_log)
    )
    decoder = SplitDecoder(
        args.model_dir,
        frames=args.frames,
        dynamic=args.dynamic,
        diagnostics=args.diagnostics,
        npu_core_mask=args.npu_core_mask,
        transformer_00_model=args.transformer_00_model,
    )
    try:
        for _ in range(args.warmup_runs):
            decoder.decode(codes)

        rows: list[dict[str, object]] = []
        last_audio: np.ndarray | None = None
        for index in range(args.runs):
            started = time.perf_counter()
            audio, codec = decoder.decode(codes)
            wall_seconds = time.perf_counter() - started
            last_audio = audio
            rows.append(
                {
                    "run": index,
                    "wall_seconds": wall_seconds,
                    "wall_rtf": wall_seconds / float(codec["audio_seconds"]),
                    "npu_seconds": codec["npu_inference_seconds"],
                    "npu_rtf": codec["codec_inference_rtf"],
                    "stage_prior_seconds": codec["stage_prior_seconds"],
                    "transformer_seconds": codec["transformer_seconds"],
                    "post_head_seconds": codec["post_head_seconds"],
                    "cpu_spectral_tail_seconds": codec["cpu_spectral_tail_seconds"],
                    "istft_seconds": codec["istft_seconds"],
                }
            )
    finally:
        decoder.close()

    wall = [float(row["wall_seconds"]) for row in rows]
    wall_rtf = [float(row["wall_rtf"]) for row in rows]
    npu = [float(row["npu_seconds"]) for row in rows]
    npu_rtf = [float(row["npu_rtf"]) for row in rows]
    stage_fields = (
        "stage_prior_seconds",
        "transformer_seconds",
        "post_head_seconds",
        "cpu_spectral_tail_seconds",
        "istft_seconds",
    )
    report = {
        "mode": "resident warm NeuCodec benchmark",
        "codes_log": str(args.codes_log) if args.codes_log else None,
        "backbone_report": str(args.backbone_report) if args.backbone_report else None,
        "input_frames": len(codes),
        "dynamic": args.dynamic,
        "diagnostics": args.diagnostics,
        "npu_core_mask": args.npu_core_mask,
        "transformer_00_model": args.transformer_00_model,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "warmup_runs": args.warmup_runs,
        "runs": args.runs,
        "wall_seconds": summarize(wall),
        "wall_rtf": summarize(wall_rtf),
        "npu_seconds": summarize(npu),
        "npu_rtf": summarize(npu_rtf),
        "stages": {
            field: summarize([float(row[field]) for row in rows])
            for field in stage_fields
        },
        "per_run": rows,
    }
    if last_audio is not None:
        report["audio_float32_sha256"] = hashlib.sha256(
            np.asarray(last_audio, dtype="<f4").tobytes()
        ).hexdigest()
    if args.output and last_audio is not None:
        write_wav(args.output, last_audio)
        report["output"] = str(args.output)
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
