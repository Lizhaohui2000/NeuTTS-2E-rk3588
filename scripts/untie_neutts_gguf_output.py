#!/usr/bin/env python3
"""Add an F16 output.weight copy to a tied-output NeuTTS GGUF model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


GGUF_PY = Path(__file__).resolve().parents[1] / "third_party" / "llama.cpp" / "gguf-py"
sys.path.insert(0, str(GGUF_PY))

import gguf  # noqa: E402


def field_value(reader: gguf.GGUFReader, key: str):
    field = reader.get_field(key)
    return field.contents() if field else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    reader = gguf.GGUFReader(args.input, "r")
    architecture = field_value(reader, gguf.Keys.General.ARCHITECTURE)
    writer = gguf.GGUFWriter(args.output, arch=architecture, endianess=reader.endianess)
    alignment = field_value(reader, gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment

    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        value_type = field.types[0]
        sub_type = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(
            field.name, field.contents(), value_type, sub_type=sub_type
        )

    token_embedding = None
    for tensor in reader.tensors:
        if tensor.name == "output.weight":
            raise ValueError("input already contains output.weight")
        if tensor.name == "token_embd.weight":
            token_embedding = tensor
        writer.add_tensor_info(
            tensor.name,
            tensor.data.shape,
            tensor.data.dtype,
            tensor.data.nbytes,
            tensor.tensor_type,
        )
    if token_embedding is None:
        raise ValueError("input does not contain token_embd.weight")
    if token_embedding.tensor_type not in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32):
        raise ValueError("input token_embd.weight must be floating point")
    writer.add_tensor_info(
        "output.weight",
        token_embedding.data.shape,
        token_embedding.data.dtype,
        token_embedding.data.nbytes,
        token_embedding.tensor_type,
    )

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    writer.write_tensor_data(token_embedding.data, tensor_endianess=reader.endianess)
    writer.close()


if __name__ == "__main__":
    main()
