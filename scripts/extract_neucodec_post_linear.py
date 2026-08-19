#!/usr/bin/env python3
"""Extract the NeuCodec post head up to the final linear STFT parameters.

RKNN 2.3.x can produce an all-zero output when the graph also contains the
final Exp/Clip/Cos/Sin operations.  The extracted graph leaves those cheap
elementwise operations for the CPU runtime.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = onnx.load(str(args.input), load_external_data=False)
    input_name = model.graph.input[0].name
    output_name = "/Transpose_output_0"
    if not any(value.name == output_name for value in model.graph.value_info):
        # The tensor is consumed by Shape/Slice nodes and may not be listed in
        # value_info in older exporters.  It is nevertheless a valid graph
        # edge, so extraction can still resolve it.
        output_name = "/Transpose_output_0"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.utils.extract_model(
        str(args.input),
        str(args.output),
        input_names=[input_name],
        output_names=[output_name],
    )
    extracted = onnx.load(str(args.output), load_external_data=False)
    print(
        f"Wrote {args.output} ({args.output.stat().st_size / 2**20:.1f} MiB), "
        f"input={input_name}, output={output_name}, nodes={len(extracted.graph.node)}"
    )


if __name__ == "__main__":
    main()
