#!/usr/bin/env python3
"""Convert a fixed-frame NeuCodec spectral decoder ONNX to RKNN."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from rknn.api import RKNN


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument(
        "--dynamic-frames",
        help="Comma-separated supported frame lengths. Builds RKNN dynamic_input "
        "variants, e.g. 256,320,384,450.",
    )
    parser.add_argument(
        "--input-name",
        help="ONNX input tensor name; inferred when the graph has one data input.",
    )
    parser.add_argument(
        "--input-shape",
        help="Comma-separated static input shape, e.g. 1,450,1024. "
        "Defaults to the normalized/codes shapes for backward compatibility.",
    )
    parser.add_argument("--target", default="rk3588")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--enable-flash-attention",
        action="store_true",
        help="enable RKNN's experimental Flash Attention graph optimization",
    )
    parser.add_argument(
        "--compress-weight",
        action="store_true",
        help="compress RKNN weights to reduce model storage",
    )
    args = parser.parse_args()

    model = onnx.load(str(args.onnx), load_external_data=False)
    initializer_names = {item.name for item in model.graph.initializer}
    graph_inputs = [
        item for item in model.graph.input if item.name not in initializer_names
    ]
    if args.input_name:
        input_value = next(
            (item for item in graph_inputs if item.name == args.input_name), None
        )
        if input_value is None:
            raise ValueError(f"input {args.input_name!r} is absent from {args.onnx}")
    elif len(graph_inputs) == 1:
        input_value = graph_inputs[0]
    else:
        raise ValueError(
            f"cannot infer one input from {[item.name for item in graph_inputs]}"
        )
    input_name = input_value.name

    if args.input_shape:
        input_shape = [int(value) for value in args.input_shape.split(",")]
        if not input_shape or any(value <= 0 for value in input_shape):
            raise ValueError(f"invalid --input-shape: {args.input_shape!r}")
    else:
        dimensions = input_value.type.tensor_type.shape.dim
        if len(dimensions) != 3:
            raise ValueError(f"expected a rank-3 input, got {len(dimensions)}")
        input_shape = []
        for index, dimension in enumerate(dimensions):
            value = dimension.dim_value
            if value:
                input_shape.append(int(value))
            elif index == 1:
                input_shape.append(args.frames)
            else:
                raise ValueError(
                    f"cannot infer dimension {index} of {input_name!r}; pass --input-shape"
                )

    converter = RKNN(verbose=args.verbose)
    try:
        dynamic_frames = None
        if args.dynamic_frames:
            lengths = [int(value) for value in args.dynamic_frames.split(",")]
            if not lengths or any(value <= 0 for value in lengths):
                raise ValueError(f"invalid --dynamic-frames: {args.dynamic_frames!r}")
            if len(set(lengths)) != len(lengths):
                raise ValueError("--dynamic-frames contains duplicate lengths")
            dynamic_frames = lengths
        ret = converter.config(
            target_platform=args.target,
            optimization_level=3,
            enable_flash_attention=args.enable_flash_attention,
            compress_weight=args.compress_weight,
            dynamic_input=(
                [
                    [[1, length, input_shape[-1]]]
                    for length in dynamic_frames
                ]
                if dynamic_frames
                else None
            ),
        )
        if ret != 0:
            raise RuntimeError(f"rknn.config failed: {ret}")
        ret = converter.load_onnx(
            model=str(args.onnx),
            inputs=[input_name],
            input_size_list=[input_shape],
        )
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx failed: {ret}")
        ret = converter.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"rknn.build failed: {ret}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        ret = converter.export_rknn(str(args.output))
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn failed: {ret}")
    finally:
        converter.release()

    print(f"Wrote {args.output} ({args.output.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
