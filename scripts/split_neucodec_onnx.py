#!/usr/bin/env python3
"""Split the exported NeuCodec spectral ONNX into RKNN-friendly stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx


PRIOR_OUTPUT = "/backbone/Transpose_1_output_0"
BLOCK_OUTPUTS = [
    f"/backbone/transformers/transformers.{index}/Add_1_output_0"
    for index in range(12)
]
POST_OUTPUT = "/Transpose_output_0"


def value_shape(model: onnx.ModelProto, name: str) -> list[int | str | None]:
    values = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    value = next((item for item in values if item.name == name), None)
    if value is None:
        return []
    return [
        dimension.dim_value or dimension.dim_param or None
        for dimension in value.type.tensor_type.shape.dim
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    model = onnx.load(str(args.input), load_external_data=False)
    graph_inputs = [
        item.name for item in model.graph.input
        if item.name not in {initializer.name for initializer in model.graph.initializer}
    ]
    if graph_inputs != ["normalized"]:
        raise ValueError(
            "expected a spectral-input export with one input named 'normalized'; "
            f"got {graph_inputs}"
        )
    required = [PRIOR_OUTPUT, *BLOCK_OUTPUTS, POST_OUTPUT]
    produced = {name for node in model.graph.node for name in node.output}
    missing = [name for name in required if name not in produced]
    if missing:
        raise ValueError(f"export does not contain expected split boundaries: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stages = [("stage_prior", "normalized", PRIOR_OUTPUT)]
    previous = PRIOR_OUTPUT
    for index, output in enumerate(BLOCK_OUTPUTS):
        stages.append((f"transformer_{index:02d}", previous, output))
        previous = output
    stages.append(("post_linear", previous, POST_OUTPUT))

    manifest = []
    for name, stage_input, stage_output in stages:
        output_path = args.output_dir / f"{name}.onnx"
        onnx.utils.extract_model(
            str(args.input),
            str(output_path),
            input_names=[stage_input],
            output_names=[stage_output],
        )
        extracted = onnx.load(str(output_path), load_external_data=False)
        manifest.append(
            {
                "name": name,
                "onnx": output_path.name,
                "input": stage_input,
                "input_shape": value_shape(extracted, stage_input),
                "output": stage_output,
                "output_shape": value_shape(extracted, stage_output),
                "nodes": len(extracted.graph.node),
                "size_bytes": output_path.stat().st_size,
            }
        )
    (args.output_dir / "manifest.json").write_text(
        json.dumps({"source": str(args.input), "stages": manifest}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
