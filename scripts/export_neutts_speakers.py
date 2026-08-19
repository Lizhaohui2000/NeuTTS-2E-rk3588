#!/usr/bin/env python3
"""Export NeuTTS fixed-speaker references to a torch-free JSON asset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    speakers = {}
    for code_path in sorted(args.samples.glob("*.pt")):
        text_path = code_path.with_suffix(".txt")
        if not text_path.exists():
            continue
        codes = torch.load(code_path, map_location="cpu", weights_only=True)
        speakers[code_path.stem] = {
            "text": text_path.read_text(encoding="utf-8").strip(),
            "codes": codes.reshape(-1).tolist(),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(speakers, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(speakers)} speakers to {args.output}")


if __name__ == "__main__":
    main()
