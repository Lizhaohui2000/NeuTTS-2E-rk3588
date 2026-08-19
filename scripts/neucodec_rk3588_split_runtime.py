#!/usr/bin/env python3
"""Run NeuTTS-2E with the NeuCodec spectral decoder split across RKNN/CPU.

The speech-token backbone remains on CPU (llama.cpp).  The token-to-spectral
path uses CPU FSQ index decoding, RK3588 NPU for the prior, twelve Transformer
blocks and post head, then a NumPy overlap-add ISTFT on CPU.

The runtime supports both the original static 450-frame graph and an optional
RKNN dynamic-shape graph with 256/320/384/450-frame variants. Generated
sequences shorter than the selected graph length are edge-padded and cropped
back to their original frame count after ISTFT; longer sequences are rejected
until chunk overlap handling is enabled explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from neutts_2e_board_runtime import EMOTIONS, run_backbone, write_wav


SAMPLE_RATE = 24_000
HOP_LENGTH = 480
N_FFT = HOP_LENGTH * 4
FFT_BINS = N_FFT // 2 + 1
SUPPORTED_DYNAMIC_FRAMES = (256, 320, 384, 450)
NPU_CORE_MASK_CHOICES = (
    "auto",
    "core0",
    "core1",
    "core2",
    "core01",
    "core012",
)


def _stats(value: np.ndarray) -> dict[str, float | bool | list[int]]:
    value = np.asarray(value)
    return {
        "shape": list(value.shape),
        "finite": bool(np.isfinite(value).all()),
        "min": float(np.nanmin(value)) if value.size else 0.0,
        "max": float(np.nanmax(value)) if value.size else 0.0,
        "rms": float(np.sqrt(np.nanmean(value * value))) if value.size else 0.0,
    }


def fsq_normalized(codes: list[int]) -> np.ndarray:
    """Decode official 8-way, four-level FSQ indices to [-1, 0.5]."""
    values = np.asarray(codes, dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("speech codes must be a non-empty 1-D sequence")
    if np.any(values < 0) or np.any(values >= 4**8):
        raise ValueError("speech codes must be in [0, 65535]")
    basis = 4 ** np.arange(8, dtype=np.int64)
    levels = (values[:, None] // basis[None, :]) % 4
    return (levels.astype(np.float32) - 2.0) / 2.0


def istft_numpy(real: np.ndarray, imag: np.ndarray) -> np.ndarray:
    """Match NeuCodec's centered Hann ISTFT for [1, 961, frames] tensors."""
    real = np.asarray(real, dtype=np.float32)
    imag = np.asarray(imag, dtype=np.float32)
    if real.ndim == 3:
        real = real[0]
    if imag.ndim == 3:
        imag = imag[0]
    if real.shape != imag.shape or real.shape[0] != FFT_BINS:
        raise ValueError(f"expected [961, frames] real/imag, got {real.shape}/{imag.shape}")

    frames = np.fft.irfft(real + 1j * imag, n=N_FFT, axis=0).astype(np.float32)
    window = np.hanning(N_FFT).astype(np.float32)
    frames *= window[:, None]
    frame_count = real.shape[1]
    full_length = N_FFT + HOP_LENGTH * (frame_count - 1)
    audio = np.zeros(full_length, dtype=np.float32)
    envelope = np.zeros(full_length, dtype=np.float32)
    window_sq = window * window
    for frame_index in range(frame_count):
        start = frame_index * HOP_LENGTH
        stop = start + N_FFT
        audio[start:stop] += frames[:, frame_index]
        envelope[start:stop] += window_sq
    pad = (N_FFT - HOP_LENGTH) // 2
    return (audio / np.maximum(envelope, 1e-8))[pad:-pad]


class SplitDecoder:
    def __init__(
        self,
        model_dir: Path,
        frames: int = 450,
        post_model: str = "post_linear.rknn",
        dynamic: bool = False,
        diagnostics: bool = False,
        npu_core_mask: str = "auto",
        transformer_00_model: str | None = None,
    ):
        try:
            from rknnlite.api import RKNNLite
        except ImportError as exc:  # pragma: no cover - exercised on board
            raise RuntimeError("rknnlite is required on the RK3588 board") from exc
        self.frames = frames
        self.dynamic = dynamic
        self.diagnostics = diagnostics
        core_mask_attributes = {
            "auto": "NPU_CORE_AUTO",
            "core0": "NPU_CORE_0",
            "core1": "NPU_CORE_1",
            "core2": "NPU_CORE_2",
            "core01": "NPU_CORE_0_1",
            "core012": "NPU_CORE_0_1_2",
        }
        try:
            core_mask = getattr(RKNNLite, core_mask_attributes[npu_core_mask])
        except KeyError as exc:
            raise ValueError(
                f"unsupported NPU core mask {npu_core_mask!r}; "
                f"choose from {NPU_CORE_MASK_CHOICES}"
            ) from exc
        self.npu_core_mask = npu_core_mask
        if dynamic:
            if frames < SUPPORTED_DYNAMIC_FRAMES[-1]:
                raise ValueError(
                    f"dynamic decoder requires max frames >= {SUPPORTED_DYNAMIC_FRAMES[-1]}, "
                    f"got {frames}"
                )
            self.supported_frames = SUPPORTED_DYNAMIC_FRAMES
        else:
            self.supported_frames = (frames,)
        self._sessions = []
        if dynamic:
            names = ["stage_prior_dynamic.rknn"] + [
                f"transformer_{i:02d}_dynamic.rknn" for i in range(12)
            ]
            if post_model == "post_linear.rknn":
                post_model = "post_linear_dynamic.rknn"
        else:
            names = ["stage_prior.rknn"] + [
                f"transformer_{i:02d}.rknn" for i in range(12)
            ]
        if transformer_00_model:
            names[1] = transformer_00_model
        names.append(post_model)
        for name in names:
            session = RKNNLite()
            ret = session.load_rknn(str(model_dir / name))
            if ret != 0:
                raise RuntimeError(f"load_rknn failed for {name}: {ret}")
            ret = session.init_runtime(core_mask=core_mask)
            if ret != 0:
                raise RuntimeError(f"init_runtime failed for {name}: {ret}")
            self._sessions.append(session)

    def close(self) -> None:
        for session in self._sessions:
            session.release()
        self._sessions.clear()

    def decode(self, codes: list[int]) -> tuple[np.ndarray, dict[str, object]]:
        original_frames = len(codes)
        if original_frames > self.frames:
            raise ValueError(
                f"generated {original_frames} frames, but static split graph supports at most "
                f"{self.frames}; use a chunked decoder for longer text"
            )
        graph_frames = next(
            (length for length in self.supported_frames if length >= original_frames),
            None,
        )
        if graph_frames is None:
            raise ValueError(
                f"generated {original_frames} frames, but supported graph lengths are "
                f"{self.supported_frames}"
            )
        normalized = fsq_normalized(codes)
        padded = np.pad(
            normalized,
            ((0, graph_frames - original_frames), (0, 0)),
            mode="edge",
        )[None]

        stage_times: list[float] = []
        started = time.perf_counter()
        hidden = self._sessions[0].inference(inputs=[padded])[0]
        stage_times.append(time.perf_counter() - started)
        hidden_stats = [_stats(hidden)] if self.diagnostics else None
        if self.diagnostics and not np.isfinite(hidden).all():
            raise FloatingPointError("stage_prior produced non-finite hidden states")

        for index, session in enumerate(self._sessions[1:13]):
            started = time.perf_counter()
            hidden = session.inference(inputs=[hidden])[0]
            stage_times.append(time.perf_counter() - started)
            if hidden_stats is not None:
                hidden_stats.append(_stats(hidden))
            if self.diagnostics and not np.isfinite(hidden).all():
                raise FloatingPointError(f"transformer_{index:02d} produced non-finite hidden states")

        started = time.perf_counter()
        outputs = self._sessions[-1].inference(inputs=[hidden])
        post_seconds = time.perf_counter() - started
        if len(outputs) != 1:
            raise RuntimeError(f"post linear head returned {len(outputs)} outputs, expected one tensor")
        linear = np.asarray(outputs[0])
        if linear.ndim != 3 or linear.shape[1] != 2 * FFT_BINS:
            raise RuntimeError(f"post linear head returned unexpected shape {linear.shape}")
        started = time.perf_counter()
        # Keep the numerically fragile trigonometric tail on CPU.  This is
        # equivalent to ISTFTHead.forward: exp(log-magnitude), clamp, phase.
        log_magnitude = linear[:, :FFT_BINS, :]
        phase = linear[:, FFT_BINS:, :]
        magnitude = np.minimum(np.exp(log_magnitude), 1e2).astype(np.float32)
        real = magnitude * np.cos(phase)
        imag = magnitude * np.sin(phase)
        spectral_seconds = time.perf_counter() - started
        output_stats = (
            {"real": _stats(real), "imag": _stats(imag)}
            if self.diagnostics
            else None
        )
        if self.diagnostics and (
            not np.isfinite(real).all() or not np.isfinite(imag).all()
        ):
            raise FloatingPointError("post head produced non-finite spectrum")

        started = time.perf_counter()
        audio = istft_numpy(real, imag)
        istft_seconds = time.perf_counter() - started
        audio = audio[: original_frames * HOP_LENGTH]
        metrics: dict[str, object] = {
            "input_frames": original_frames,
            "padded_frames": graph_frames,
            "dynamic_graph": self.dynamic,
            "stage_prior_seconds": stage_times[0],
            "transformer_seconds": sum(stage_times[1:]),
            "transformer_seconds_each": stage_times[1:],
            "post_head_seconds": post_seconds,
            "cpu_spectral_tail_seconds": spectral_seconds,
            "istft_seconds": istft_seconds,
            "npu_inference_seconds": sum(stage_times) + post_seconds,
            "audio_seconds": float(audio.size / SAMPLE_RATE),
        }
        if hidden_stats is not None:
            metrics["hidden_stats"] = hidden_stats
        if output_stats is not None:
            metrics["spectrum_stats"] = output_stats
        if self.diagnostics:
            metrics.update(
                {
                    "audio_peak": float(np.max(np.abs(audio))) if audio.size else 0.0,
                    "audio_rms": float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0,
                    "audio_finite": bool(np.isfinite(audio).all()),
                }
            )
        metrics["codec_inference_rtf"] = float(metrics["npu_inference_seconds"]) / max(
            float(metrics["audio_seconds"]), 1e-9
        )
        return audio, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-cli", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
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
    parser.add_argument("--emotion", choices=EMOTIONS, default="happy")
    parser.add_argument("--text", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--predict", type=int, default=450)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-threads", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--speech-only-head",
        action="store_true",
        help="enable the patched NeuTTS 65K speech-token output projection",
    )
    parser.add_argument("--frames", type=int, default=450)
    parser.add_argument("--post-model", default="post_linear.rknn")
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="use stage_prior/Transformer/post-linear RKNN dynamic-shape variants",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="collect per-layer tensor statistics; disabled by default for production RTF",
    )
    parser.add_argument(
        "--npu-core-mask",
        choices=NPU_CORE_MASK_CHOICES,
        default="auto",
        help="RKNN NPU cores available to every serial decoder stage",
    )
    parser.add_argument(
        "--transformer-00-model",
        help="optional model filename replacing Transformer block 0 for isolated A/B tests",
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
    emotion_token = f"<|{args.emotion.upper()}|>"
    reference_tokens = "".join(f"<|speech_{code}|>" for code in reference_codes)
    prompt = (
        f"<|TEXT_PROMPT_START|>{reference_text}{emotion_token}{args.text}"
        f"<|TEXT_PROMPT_END|><|SPEECH_GENERATION_START|>{reference_tokens}"
    )
    codes, backbone_metrics, raw_output = run_backbone(
        args.llama_cli,
        args.model,
        prompt,
        len(reference_codes),
        args.predict,
        args.threads,
        args.batch_threads,
        args.temperature,
        args.top_k,
        args.seed,
        args.speech_only_head,
    )
    decoder = SplitDecoder(
        args.model_dir,
        frames=args.frames,
        post_model=args.post_model,
        dynamic=args.dynamic,
        diagnostics=args.diagnostics,
        npu_core_mask=args.npu_core_mask,
        transformer_00_model=args.transformer_00_model,
    )
    try:
        decoder_started = time.perf_counter()
        audio, codec_metrics = decoder.decode(codes)
        codec_metrics["model_load_rss_mib"] = None
        codec_metrics["decode_wall_seconds"] = time.perf_counter() - decoder_started
    finally:
        decoder.close()

    write_wav(args.output, audio)
    audio_seconds = float(audio.size / SAMPLE_RATE)
    total_seconds = float(backbone_metrics["backbone_wall_seconds"]) + float(
        codec_metrics["decode_wall_seconds"]
    )
    metrics = {
        "model": "NeuTTS-2E Q4_K_M + NeuCodec RKNN split + CPU ISTFT",
        "speaker": args.speaker,
        "reference_codes": len(reference_codes),
        "reference_text": reference_text,
        "emotion": args.emotion,
        "text": args.text,
        "speech_only_head": args.speech_only_head,
        "npu_core_mask": args.npu_core_mask,
        "transformer_00_model": args.transformer_00_model,
        "output": str(args.output),
        "sample_rate": SAMPLE_RATE,
        "generated_speech_tokens": len(codes),
        **backbone_metrics,
        **codec_metrics,
        "audio_seconds": audio_seconds,
        "end_to_end_seconds": total_seconds,
        "end_to_end_rtf": total_seconds / max(audio_seconds, 1e-9),
    }
    metrics_path = args.metrics or args.output.with_suffix(".json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".llama.log").write_bytes(raw_output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout.flush()
