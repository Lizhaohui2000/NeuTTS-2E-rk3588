# Complete-context short-reference study

This document records the experiment originally called the **Natural-boundary 103 study** and explains how its result affects the release design.

## Question

The original 103-code prefix was fast for most evaluated conditions but produced severe Angry intelligibility failures. There were two possible explanations:

1. Angry inherently needs a longer reference.
2. The selected short reference is linguistically or prosodically incomplete.

The prefix ends at a real low-energy word boundary, but its transcript is an unfinished phrase: *“What we need, helping us to develop”*. Low energy alone therefore does not establish a good prompt boundary.

## Protocol

The host screen used NeuTTS-2E Q4_K_M with the FP32 ONNX NeuCodec decoder. It covered Emily, Angry, three English texts and seeds 42, 123 and 2026. Every condition used temperature 1.0, top-k 50 and a 450-code generation limit.

Four references were compared:

- Prefix `[0,103)`: the original incomplete phrase.
- Natural slice `[103,206)`: an equal-length slice aligned with the complete phrase *“the scope of work for a future procurement.”*
- Re-encoded natural crop: the same 2.06-second phrase cropped from the waveform and independently NeuCodec-encoded.
- Prefix `[0,207)`: the stable full reference.

## Results

| Reference | Codes | Mean prompt tokens | Micro WER | WER > 20% | Mean output codes | Angry probability | DNSMOS SIG / OVR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Prefix `[0,103)` | 103 | 129.3 | 34.2% | 6/9 | 305.2 | **0.775** | 4.041 / 4.001 |
| Natural slice `[103,206)` | 103 | 130.3 | 6.1% | 0/9 | 280.4 | 0.751 | 4.024 / 3.914 |
| Independently re-encoded crop | 103 | 130.3 | **2.6%** | **0/9** | 276.9 | 0.745 | **4.070 / 4.036** |
| Stable prefix `[0,207)` | 207 | 242.3 | **2.6%** | **0/9** | 279.2 | 0.741 | 3.951 / 3.925 |

The independently encoded crop agrees position-by-position with the sliced sequence at only 25.2%, yet matches the 207-code reference WER. In this experiment, the original Angry failure is therefore best explained by incomplete linguistic/prosodic reference context rather than by a requirement for 207 codes or dependence on a special token slice.

This is a limited result, not a universal claim. It contains one speaker, one emotion, three texts and three seeds. It does not establish a universal duration, prove that every complete clause works, or provide a multi-speaker RK3588 RTF result.

Raw metrics are available in [natural_boundary_103_host.json](../results/natural_boundary_103_host.json) and [natural_boundary_103_reencoded_control.json](../results/natural_boundary_103_reencoded_control.json).

## Same-text audio control

Text, speaker and seed are identical across these examples.

| Prefix 103 | Natural 103 slice | Re-encoded natural 103 | Prefix 207 |
|---|---|---|---|
| [listen](../samples/reference_boundary/prefix103_angry.wav?raw=1) | [listen](../samples/reference_boundary/natural103_angry.wav?raw=1) | [listen](../samples/reference_boundary/natural103_reencoded_angry.wav?raw=1) | [listen](../samples/reference_boundary/prefix207_angry.wav?raw=1) |

## Release decision

The experiment changes the mechanism used by the release without promoting an unverified result to the default:

- `stable` always selects the 207-code reference.
- `fast` selects its configured short reference when it is marked `release_eligible` for the requested emotion.
- If the short reference is not validated for that emotion, `fast` automatically falls back to the stable reference.
- `experimental_natural103_slice` reproduces the sliced control.
- `experimental_reencoded103` loads the independently encoded codes from [emily_natural103_reencoded.json](../configs/emily_natural103_reencoded.json).

The old names remain CLI aliases, but `natural103` deliberately maps to the sliced experimental control. This avoids claiming that the slice and independent re-encoding are the same condition.

## Profile contract

Reference policy is configured per speaker in [reference_strategies.json](../configs/reference_strategies.json). A short release candidate should contain:

```json
{
  "source": "code_file",
  "path": "speaker_short_reference.json",
  "start": 0,
  "count": 103,
  "text": "An exact transcript of one complete phrase.",
  "release_eligible": true,
  "validated_emotions": ["neutral", "happy", "sad"],
  "validation_status": "RK3588 board-evaluated"
}
```

The standalone file is JSON with a `codes` array. `speaker_codes` may instead select a range from the official exported speaker asset. A release mode with `require_release_eligible` falls back when the requested emotion is absent from `validated_emotions`, and fails closed when no release-eligible fallback exists.

## Calibration protocol for another speaker

Reference compilation should happen offline; it adds no board-side model or runtime computation.

1. Obtain a clean reference waveform and exact transcript.
2. Align words or phonemes and find candidates near 1.8 to 2.5 seconds.
3. Require low-energy acoustic boundaries, a complete clause or prosodic constituent, sufficient voiced content and no clipped word.
4. Crop each candidate waveform and independently NeuCodec-encode it. Do not assume a slice from a longer code stream is equivalent.
5. Evaluate candidates on held-out texts and seeds for every intended emotion.
6. Compare against the stable reference and mark only passing emotions in `validated_emotions`.
7. Measure the final policy on the target RK3588 configuration.

Reasonable engineering gates are 100% generation completion, WER no more than two percentage points above the stable reference, DNSMOS SIG loss no greater than 0.1, speaker-similarity loss no greater than 0.02, no meaningful emotion-target degradation and a measured board-side RTF benefit. These are proposed release gates, not conclusions established by the small Emily experiment.

An automatic aligner and quality evaluator are intentionally not bundled into the board runtime. They require heavier ASR, alignment, speaker and quality models and belong in an offline calibration toolchain. The runtime consumes only the accepted codes, transcript and validation metadata.
