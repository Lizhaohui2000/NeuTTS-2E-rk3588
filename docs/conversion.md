# NeuCodec RKNN conversion

The repository does not contain model weights, ONNX graphs or RKNN binaries. Generate them from an officially downloaded NeuCodec checkpoint.

Host requirements:

- Python environment containing the official NeuTTS/NeuCodec implementation
- PyTorch, ONNX and ONNX Runtime
- Rockchip RKNN Toolkit2 compatible with the target board runtime

## Export spectral graph

```bash
PYTHONPATH=/path/to/neutts:$PYTHONPATH \
python scripts/export_neucodec_decoder.py \
  --checkpoint /path/to/neucodec/model.safetensors-or-checkpoint.pt \
  --output artifacts/neucodec-spectral.onnx \
  --frames 450 \
  --spectral-output \
  --spectral-input
```

The exporter expects the checkpoint structure used by the official NeuCodec release. Adjust the checkpoint filename to the downloaded asset.

## Split RKNN-friendly stages

```bash
python scripts/split_neucodec_onnx.py \
  --input artifacts/neucodec-spectral.onnx \
  --output-dir artifacts/stages
```

This produces `stage_prior.onnx`, twelve `transformer_XX.onnx` graphs, `post_linear.onnx` and a manifest. Exp/Cos/Sin and ISTFT intentionally remain on CPU because they were numerically fragile under the tested RKNN 2.3.x toolchain.

## Convert dynamic RKNN models

```bash
for model in stage_prior transformer_{00..11} post_linear; do
  python scripts/convert_neucodec_rknn.py \
    --onnx "artifacts/stages/${model}.onnx" \
    --output "models_dynamic/${model}_dynamic.rknn" \
    --frames 450 \
    --dynamic-frames 256,320,384,450 \
    --target rk3588
done
```

Copy `models_dynamic/` to the RK3588. The runtime chooses the smallest supported graph that can contain the generated speech-code sequence, pads only for RKNN invocation and crops the waveform to the original code length.

Verify audio quality and numerical validity before deploying artifacts produced by a different RKNN Toolkit/Lite version.
