#!/usr/bin/env python3
"""Export the NeuCodec token-to-waveform decoder without loading its encoder."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class RealISTFT(nn.Module):
    """ONNX-friendly replacement for torch.fft.irfft plus overlap-add."""

    def __init__(self, n_fft: int, hop_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.pad = (n_fft - hop_length) // 2

        n = torch.arange(n_fft, dtype=torch.float32)[:, None]
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float32)[None, :]
        angle = 2.0 * math.pi * n * k / n_fft
        scale = torch.full((n_fft // 2 + 1,), 2.0 / n_fft)
        scale[0] = 1.0 / n_fft
        scale[-1] = 1.0 / n_fft
        self.register_buffer("cos_basis", torch.cos(angle) * scale)
        self.register_buffer("sin_basis", torch.sin(angle) * scale)
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        frames = torch.matmul(self.cos_basis, real) - torch.matmul(self.sin_basis, imag)
        frames = frames * self.window[None, :, None]

        channels = self.n_fft
        ola_weight = torch.eye(channels, device=frames.device, dtype=frames.dtype)
        ola_weight = ola_weight.reshape(channels, 1, channels)
        audio = F.conv_transpose1d(frames, ola_weight, stride=self.hop_length)[:, 0]

        frame_count = real.shape[-1]
        window_frames = self.window.square()[None, :, None].expand(1, -1, frame_count)
        envelope = F.conv_transpose1d(window_frames, ola_weight, stride=self.hop_length)[:, 0]
        return (audio / envelope.clamp_min(1e-8))[:, self.pad : -self.pad]


class DecoderOnly(nn.Module):
    def __init__(self, generator: nn.Module, fc_post_a: nn.Module):
        super().__init__()
        self.project_out = generator.quantizer.project_out
        self.backbone = generator.backbone
        self.head = generator.head
        self.fc_post_a = fc_post_a
        self.register_buffer("fsq_basis", 4 ** torch.arange(8, dtype=torch.long))
        old_head = self.head
        self.head.istft = RealISTFT(
            old_head.istft.n_fft, old_head.istft.hop_length
        )

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        indices = codes[:, 0, :, None].to(self.project_out.weight.dtype)
        basis = self.fsq_basis.to(self.project_out.weight.dtype)
        # ResidualFSQ.indices_to_level_indices: (indices // basis) % levels.
        # Omitting the modulo leaves large higher-order digits (e.g. 4095),
        # which overflows the NPU graph for otherwise valid 16-bit codes.
        levels = torch.remainder(torch.floor(indices / basis), 4.0)
        normalized = (levels - 2.0) / 2.0
        emb = self.project_out(normalized)
        emb = self.fc_post_a(emb)
        hidden = self.backbone(emb)
        x_pred = self.head.out(hidden).transpose(1, 2)
        mag, phase = x_pred.chunk(2, dim=1)
        mag = torch.exp(mag).clamp(max=1e2)
        audio = self.head.istft(mag * torch.cos(phase), mag * torch.sin(phase))
        return audio.unsqueeze(1)


class SpectralDecoder(nn.Module):
    """Decode speech codes to the real and imaginary STFT components."""

    def __init__(self, generator: nn.Module, fc_post_a: nn.Module):
        super().__init__()
        self.project_out = generator.quantizer.project_out
        self.backbone = generator.backbone
        self.head_out = generator.head.out
        self.fc_post_a = fc_post_a
        self.register_buffer("fsq_basis", 4 ** torch.arange(8, dtype=torch.long))

    def forward(self, codes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        indices = codes[:, 0, :, None].to(self.project_out.weight.dtype)
        basis = self.fsq_basis.to(self.project_out.weight.dtype)
        levels = torch.remainder(torch.floor(indices / basis), 4.0)
        normalized = (levels - 2.0) / 2.0
        emb = self.project_out(normalized)
        emb = self.fc_post_a(emb)
        hidden = self.backbone(emb)
        x_pred = self.head_out(hidden).transpose(1, 2)
        mag, phase = x_pred.chunk(2, dim=1)
        mag = torch.exp(mag).clamp(max=1e2)
        return mag * torch.cos(phase), mag * torch.sin(phase)


class SpectralDecoderInput(nn.Module):
    """Spectral decoder whose boundary is the float FSQ levels tensor."""

    def __init__(self, generator: nn.Module, fc_post_a: nn.Module):
        super().__init__()
        self.project_out = generator.quantizer.project_out
        self.backbone = generator.backbone
        self.head_out = generator.head.out
        self.fc_post_a = fc_post_a

    def forward(self, normalized: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        emb = self.project_out(normalized)
        emb = self.fc_post_a(emb)
        hidden = self.backbone(emb)
        x_pred = self.head_out(hidden).transpose(1, 2)
        mag, phase = x_pred.chunk(2, dim=1)
        mag = torch.exp(mag).clamp(max=1e2)
        return mag * torch.cos(phase), mag * torch.sin(phase)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--verify-reference", type=Path)
    parser.add_argument(
        "--spectral-output",
        action="store_true",
        help="Export real/imaginary STFT tensors and leave ISTFT to the CPU runtime.",
    )
    parser.add_argument(
        "--spectral-input",
        action="store_true",
        help="Use float FSQ levels [batch, frames, 8] as the spectral graph input.",
    )
    args = parser.parse_args()

    from neucodec.codec_decoder_vocos import CodecDecoderVocos

    print("Constructing decode-only NeuCodec graph", flush=True)
    generator = CodecDecoderVocos(hop_length=480)
    generator.apply_weight_norm()
    fc_post_a = nn.Linear(2048, 1024)

    print(f"Reading decoder weights from {args.checkpoint}", flush=True)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    generator_state = {
        key.removeprefix("generator."): value
        for key, value in state.items()
        if key.startswith("generator.")
        and not (
            key.endswith(".weight")
            and f"{key[:-7]}.weight_g" in state
            and f"{key[:-7]}.weight_v" in state
        )
    }
    fc_state = {
        key.removeprefix("fc_post_a."): value
        for key, value in state.items()
        if key.startswith("fc_post_a.")
    }
    generator.load_state_dict(generator_state, strict=True)
    fc_post_a.load_state_dict(fc_state, strict=True)
    del state, generator_state, fc_state
    generator.remove_weight_norm()

    codes = torch.zeros((1, 1, args.frames), dtype=torch.long)
    if args.verify_reference:
        ref = torch.load(args.verify_reference, map_location="cpu", weights_only=True).long()
        codes = ref.reshape(1, 1, -1)[..., : args.frames]

    generator.eval()
    fc_post_a.eval()
    print("Computing the original complex-ISTFT reference", flush=True)
    with torch.inference_mode():
        emb = generator.quantizer.get_output_from_indices(codes.transpose(1, 2))
        original_waveform = generator(fc_post_a(emb), vq=False)[0]

    decoder = DecoderOnly(generator, fc_post_a).eval()
    print("Checking the real-valued ISTFT replacement", flush=True)
    with torch.inference_mode():
        waveform = decoder(codes)
    expected_samples = args.frames * 480
    assert waveform.shape == (1, 1, expected_samples), waveform.shape
    assert torch.isfinite(waveform).all()
    max_error = (waveform - original_waveform).abs().max().item()
    mean_error = (waveform - original_waveform).abs().mean().item()
    assert max_error < 1e-4, max_error
    print(
        f"PyTorch check: codes={tuple(codes.shape)} waveform={tuple(waveform.shape)} "
        f"peak={waveform.abs().max().item():.6f} max_error={max_error:.3e} "
        f"mean_error={mean_error:.3e}",
        flush=True,
    )

    export_model: nn.Module = decoder
    output_names = ["audio"]
    export_input = codes
    input_name = "codes"
    dynamic_axes = {"codes": {2: "frames"}, "audio": {2: "samples"}}
    if args.spectral_output:
        spectral_decoder = (
            SpectralDecoderInput(generator, fc_post_a)
            if args.spectral_input
            else SpectralDecoder(generator, fc_post_a)
        ).eval()
        if args.spectral_input:
            indices = codes[:, 0, :, None].to(generator.quantizer.project_out.weight.dtype)
            basis = (4 ** torch.arange(8, dtype=torch.long)).to(indices.dtype)
            levels = torch.remainder(torch.floor(indices / basis), 4.0)
            export_input = (levels - 2.0) / 2.0
            input_name = "normalized"
        with torch.inference_mode():
            real, imag = spectral_decoder(export_input)
            spectral_waveform = decoder.head.istft(real, imag).unsqueeze(1)
        spectral_error = (spectral_waveform - waveform).abs().max().item()
        assert spectral_error < 1e-6, spectral_error
        export_model = spectral_decoder
        output_names = ["real", "imag"]
        dynamic_axes = {
            input_name: {1: "frames"} if args.spectral_input else {2: "frames"},
            "real": {2: "frames"},
            "imag": {2: "frames"},
        }
        print(
            f"Spectral split check: real={tuple(real.shape)} imag={tuple(imag.shape)} "
            f"waveform_error={spectral_error:.3e}",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting ONNX opset {args.opset} to {args.output}", flush=True)
    torch.onnx.export(
        export_model,
        (export_input,),
        str(args.output),
        input_names=[input_name],
        output_names=output_names,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )
    print(f"Wrote {args.output} ({args.output.stat().st_size / 2**20:.1f} MiB)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"export failed: {exc}", file=sys.stderr)
        raise
