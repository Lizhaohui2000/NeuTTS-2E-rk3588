#!/usr/bin/env python3
"""Minimal torch-free NeuTTS-2E runner for an ARM64 Linux board."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort


SPEECH_TOKEN = re.compile(rb"<\|speech_(\d+)\|>")
EMOTIONS = ("neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised")


def peak_rss_monitor(pid: int, stop: threading.Event, result: list[int]) -> None:
    peak_kib = 0
    status_path = Path(f"/proc/{pid}/status")
    while not stop.wait(0.02):
        try:
            status = status_path.read_text(encoding="utf-8")
        except (FileNotFoundError, ProcessLookupError):
            break
        match = re.search(r"^VmRSS:\s+(\d+)\s+kB$", status, re.MULTILINE)
        if match:
            peak_kib = max(peak_kib, int(match.group(1)))
    result.append(peak_kib)


def current_rss_mib() -> float:
    status = Path("/proc/self/status").read_text(encoding="utf-8")
    match = re.search(r"^VmRSS:\s+(\d+)\s+kB$", status, re.MULTILINE)
    return int(match.group(1)) / 1024.0 if match else 0.0


def run_backbone(
    llama_cli: Path,
    model: Path,
    prompt: str,
    reference_count: int,
    predict: int,
    threads: int,
    batch_threads: int,
    temperature: float,
    top_k: int,
    seed: int,
    speech_only_head: bool = False,
) -> tuple[list[int], dict[str, float | int | None], bytes]:
    command = [
        str(llama_cli),
        "-m",
        str(model),
        "-p",
        prompt,
        "-n",
        str(predict),
        "-c",
        "2048",
        "-t",
        str(threads),
        "-tb",
        str(batch_threads),
        "--temp",
        str(temperature),
        "--top-k",
        str(top_k),
        "--seed",
        str(seed),
        "-r",
        "<|SPEECH_GENERATION_END|>",
        "-no-cnv",
        "-st",
        "-sp",
        "--no-display-prompt",
        "--no-show-timings",
        "--log-disable",
    ]
    started = time.perf_counter()
    environment = os.environ.copy()
    if speech_only_head:
        environment["LLAMA_NEUTTS_SPEECH_ONLY"] = "1"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
    )
    stop_monitor = threading.Event()
    peak_rss: list[int] = []
    monitor = threading.Thread(
        target=peak_rss_monitor, args=(process.pid, stop_monitor, peak_rss), daemon=True
    )
    monitor.start()

    output = bytearray()
    token_timestamps: list[float] = []
    parsed_count = 0
    assert process.stdout is not None
    while True:
        chunk = os.read(process.stdout.fileno(), 4096)
        if not chunk:
            break
        output.extend(chunk)
        found = list(SPEECH_TOKEN.finditer(output))
        if len(found) > parsed_count:
            now = time.perf_counter()
            token_timestamps.extend([now] * (len(found) - parsed_count))
            parsed_count = len(found)

    return_code = process.wait()
    stop_monitor.set()
    monitor.join()
    finished = time.perf_counter()
    if return_code != 0:
        raise RuntimeError(f"llama-cli exited with {return_code}: {bytes(output[-2000:])!r}")

    marker = b"<|SPEECH_GENERATION_START|>"
    marker_index = output.rfind(marker)
    if marker_index < 0:
        raise RuntimeError("llama-cli output did not contain the speech generation marker")
    marker_region = bytes(output[marker_index:])
    truncation_marker = b"... (truncated)\n"
    if truncation_marker in marker_region:
        generation_region = marker_region.split(truncation_marker, 1)[1]
    else:
        encoded_prompt = prompt.encode("utf-8")
        prompt_index = output.find(encoded_prompt)
        generation_region = (
            bytes(output[prompt_index + len(encoded_prompt) :])
            if prompt_index >= 0
            else marker_region
        )
    generation_region = generation_region.split(b"<|SPEECH_GENERATION_END|>", 1)[0]
    generated_codes = [int(match.group(1)) for match in SPEECH_TOKEN.finditer(generation_region)]
    if not generated_codes:
        raise RuntimeError(
            "llama-cli did not generate any speech tokens; "
            f"reference_count={reference_count} "
            f"after_marker={bytes(output[marker_index:marker_index + 1000])!r} "
            f"tail={bytes(output[-1000:])!r}"
        )

    generated_timestamps = token_timestamps[-len(generated_codes) :]
    first_token_seconds = generated_timestamps[0] - started if generated_timestamps else None
    generation_window = (
        generated_timestamps[-1] - generated_timestamps[0]
        if len(generated_timestamps) > 1
        else 0.0
    )
    metrics = {
        "backbone_wall_seconds": finished - started,
        "time_to_first_speech_token_seconds": first_token_seconds,
        "generated_speech_tokens": len(generated_codes),
        "steady_speech_tokens_per_second": (
            (len(generated_timestamps) - 1) / generation_window if generation_window > 0 else None
        ),
        "backbone_peak_rss_mib": max(peak_rss, default=0) / 1024.0,
    }
    return generated_codes, metrics, bytes(output)


def decode(codec: Path, codes: list[int], threads: int) -> tuple[np.ndarray, dict[str, float]]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    rss_before = current_rss_mib()
    started = time.perf_counter()
    session = ort.InferenceSession(
        str(codec), sess_options=options, providers=["CPUExecutionProvider"]
    )
    load_seconds = time.perf_counter() - started
    rss_after_load = current_rss_mib()
    code_array = np.asarray(codes, dtype=np.int64).reshape(1, 1, -1)
    started = time.perf_counter()
    audio = session.run(None, {"codes": code_array})[0].reshape(-1).astype(np.float32)
    decode_seconds = time.perf_counter() - started
    rss_after_decode = current_rss_mib()
    return audio, {
        "codec_load_seconds": load_seconds,
        "codec_inference_seconds": decode_seconds,
        "codec_rss_increase_mib": max(rss_after_load, rss_after_decode) - rss_before,
        "codec_process_rss_mib": max(rss_after_load, rss_after_decode),
    }


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = 24_000) -> None:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    gain = min(1.0, 0.8912509 / max(peak, 1e-9))
    pcm = np.clip(audio * gain * 32767.0, -32768, 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-cli", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--codec", type=Path, required=True)
    parser.add_argument("--speakers", type=Path, required=True)
    parser.add_argument("--speaker", default="emily")
    parser.add_argument("--emotion", choices=EMOTIONS, default="happy")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--predict", type=int, default=400)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-threads", type=int, default=8)
    parser.add_argument("--codec-threads", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--speech-only-head",
        action="store_true",
        help="enable the patched NeuTTS 65K speech-token output projection",
    )
    args = parser.parse_args()

    speakers = json.loads(args.speakers.read_text(encoding="utf-8"))
    reference = speakers[args.speaker]
    emotion_token = f"<|{args.emotion.upper()}|>"
    reference_tokens = "".join(f"<|speech_{code}|>" for code in reference["codes"])
    prompt = (
        f"<|TEXT_PROMPT_START|>{reference['text']}{emotion_token}{args.text}"
        f"<|TEXT_PROMPT_END|><|SPEECH_GENERATION_START|>{reference_tokens}"
    )

    codes, backbone_metrics, raw_output = run_backbone(
        args.llama_cli,
        args.model,
        prompt,
        len(reference["codes"]),
        args.predict,
        args.threads,
        args.batch_threads,
        args.temperature,
        args.top_k,
        args.seed,
        args.speech_only_head,
    )
    audio, codec_metrics = decode(args.codec, codes, args.codec_threads)
    write_wav(args.output, audio)
    audio_seconds = audio.size / 24_000.0
    codec_seconds = codec_metrics["codec_load_seconds"] + codec_metrics["codec_inference_seconds"]
    total_seconds = float(backbone_metrics["backbone_wall_seconds"]) + codec_seconds
    codec_label = "INT8" if "int8" in args.codec.name.lower() else "FP32"
    metrics = {
        "model": f"NeuTTS-2E Q4_K_M + NeuCodec ONNX {codec_label}",
        "codec": codec_label,
        "speaker": args.speaker,
        "emotion": args.emotion,
        "text": args.text,
        "speech_only_head": args.speech_only_head,
        **backbone_metrics,
        **codec_metrics,
        "codec_total_seconds_including_load": codec_seconds,
        "codec_inference_rtf": codec_metrics["codec_inference_seconds"]
        / max(audio_seconds, 1e-9),
        "audio_seconds": audio_seconds,
        "end_to_end_seconds": total_seconds,
        "end_to_end_rtf": total_seconds / max(audio_seconds, 1e-9),
        "sample_rate": 24_000,
        "output": str(args.output),
    }
    metrics_path = args.metrics or args.output.with_suffix(".json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".llama.log").write_bytes(raw_output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
