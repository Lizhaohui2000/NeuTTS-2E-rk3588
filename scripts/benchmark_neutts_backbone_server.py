#!/usr/bin/env python3
"""Benchmark a resident llama-server NeuTTS backbone after warm-up."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


EMOTIONS = ("neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised")
SPEECH_TOKEN = re.compile(r"<\|speech_(\d+)\|>")
SPEECH_FIRST_ID = 151684
SPEECH_TOKEN_COUNT = 65536


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty list")
    position = (len(ordered) - 1) * quantile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
    }


def first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (left_code, right_code) in enumerate(zip(left, right)):
        if left_code != right_code:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def request_json(url: str, path: str, timeout: float) -> object:
    with urllib.request.urlopen(f"{url.rstrip('/')}{path}", timeout=timeout) as response:
        return json.loads(response.read())


def completion(
    url: str,
    prompt: str,
    predict: int,
    temperature: float,
    top_k: int,
    seed: int,
    timeout: float,
    cache_prompt: bool = False,
) -> dict[str, object]:
    payload = {
        "prompt": prompt,
        "n_predict": predict,
        "temperature": temperature,
        "top_k": top_k,
        "samplers": ["top_k", "temperature"],
        "seed": seed,
        "stop": ["<|SPEECH_GENERATION_END|>"],
        "stream": True,
        "cache_prompt": cache_prompt,
        "return_tokens": True,
    }
    request = urllib.request.Request(
        f"{url.rstrip('/')}/completion",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    content: list[str] = []
    token_ids: list[int] = []
    final: dict[str, object] | None = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                continue
            event = json.loads(data)
            content.append(str(event.get("content", "")))
            event_tokens = event.get("tokens")
            if isinstance(event_tokens, list):
                token_ids.extend(int(token) for token in event_tokens)
            if event.get("stop"):
                final = event
    wall_seconds = time.perf_counter() - started
    if final is None:
        raise RuntimeError("llama-server stream ended without a final timing event")
    timings = final.get("timings")
    if not isinstance(timings, dict):
        raise RuntimeError("llama-server final event has no timings object")
    predicted_per_second = float(timings["predicted_per_second"])
    generated = "".join(content)
    rendered_codes = [int(value) for value in SPEECH_TOKEN.findall(generated)]
    speech_codes = [
        token_id - SPEECH_FIRST_ID
        for token_id in token_ids
        if SPEECH_FIRST_ID <= token_id < SPEECH_FIRST_ID + SPEECH_TOKEN_COUNT
    ]
    if rendered_codes and rendered_codes != speech_codes:
        raise RuntimeError("rendered and raw-ID NeuTTS speech tokens disagree")
    if not speech_codes:
        raise RuntimeError("completion did not contain any NeuTTS speech tokens")
    code_bytes = ",".join(str(code) for code in speech_codes).encode("ascii")
    return {
        "wall_seconds": wall_seconds,
        "prompt_n": int(timings.get("prompt_n", 0)),
        "prompt_ms": float(timings.get("prompt_ms", 0.0)),
        "prompt_per_second": float(timings.get("prompt_per_second", 0.0)),
        "predicted_n": int(timings["predicted_n"]),
        "predicted_ms": float(timings["predicted_ms"]),
        "predicted_per_second": predicted_per_second,
        "steady_backbone_rtf": 50.0 / predicted_per_second,
        "speech_tokens": len(speech_codes),
        "speech_code_sha256": hashlib.sha256(code_bytes).hexdigest(),
        "speech_codes": speech_codes,
        "server_token_ids": token_ids,
    }


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
        help="reuse the resident server KV prefix between repeated requests",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--anchor",
        type=Path,
        help="resident benchmark report whose first run is the token anchor",
    )
    parser.add_argument(
        "--codes-output",
        type=Path,
        help="write the first measured speech-token sequence in llama text form",
    )
    parser.add_argument(
        "--configuration",
        default="unspecified",
        help="label recorded in the report, e.g. q4_0+speech_only+kleidiai",
    )
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
    emotion_token = f"<|{args.emotion.upper()}|>"
    prompt = (
        f"<|TEXT_PROMPT_START|>{reference_text}{emotion_token}{args.text}"
        f"<|TEXT_PROMPT_END|><|SPEECH_GENERATION_START|>{reference_tokens}"
    )

    try:
        health = request_json(args.url, "/health", args.timeout)
        props = request_json(args.url, "/props", args.timeout)
        for _ in range(args.warmup_runs):
            completion(
                args.url,
                prompt,
                args.predict,
                args.temperature,
                args.top_k,
                args.seed,
                args.timeout,
                args.cache_prompt,
            )
        rows = [
            completion(
                args.url,
                prompt,
                args.predict,
                args.temperature,
                args.top_k,
                args.seed,
                args.timeout,
                args.cache_prompt,
            )
            for _ in range(args.runs)
        ]
    except urllib.error.URLError as exc:
        raise SystemExit(f"cannot benchmark llama-server at {args.url}: {exc}") from exc

    first_codes = rows[0]["speech_codes"]
    agreement = [row["speech_codes"] == first_codes for row in rows]
    report = {
        "mode": "resident warm NeuTTS backbone benchmark",
        "configuration": args.configuration,
        "server_url": args.url,
        "server_health": health,
        "server_props": props,
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
        "warmup_runs": args.warmup_runs,
        "runs": args.runs,
        "predicted_per_second": summarize(
            [float(row["predicted_per_second"]) for row in rows]
        ),
        "steady_backbone_rtf": summarize(
            [float(row["steady_backbone_rtf"]) for row in rows]
        ),
        "wall_seconds": summarize([float(row["wall_seconds"]) for row in rows]),
        "prompt_ms": summarize([float(row["prompt_ms"]) for row in rows]),
        "all_token_sequences_identical": all(agreement),
        "token_sequence_agreement": agreement,
        "per_run": rows,
    }
    if args.anchor:
        anchor_report = json.loads(args.anchor.read_text(encoding="utf-8"))
        anchor_codes = anchor_report["per_run"][0]["speech_codes"]
        overlap = min(len(anchor_codes), len(first_codes))
        same = sum(
            anchor_codes[index] == first_codes[index] for index in range(overlap)
        )
        report["anchor"] = {
            "report": str(args.anchor),
            "speech_code_sha256": anchor_report["per_run"][0][
                "speech_code_sha256"
            ],
            "exact_match": anchor_codes == first_codes,
            "first_divergence": first_divergence(anchor_codes, first_codes),
            "position_agreement": same / max(overlap, 1),
            "overlap_tokens": overlap,
        }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.codes_output:
        args.codes_output.parent.mkdir(parents=True, exist_ok=True)
        args.codes_output.write_text(
            "<|SPEECH_GENERATION_START|>"
            + "".join(f"<|speech_{code}|>" for code in first_codes)
            + "<|SPEECH_GENERATION_END|>\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
