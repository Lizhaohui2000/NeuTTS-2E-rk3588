from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_neutts_backbone_server import select_reference_window  # noqa: E402


def test_selects_an_equal_length_shifted_window():
    codes, text, end = select_reference_window(
        [1, 2, 3, 4], "full", start=1, limit=2, window_text="middle"
    )
    assert codes == [2, 3]
    assert text == "middle"
    assert end == 3


def test_window_requires_a_matching_transcript():
    try:
        select_reference_window([1, 2, 3], "full", start=1, limit=1)
    except ValueError as error:
        assert "reference-text" in str(error)
    else:
        raise AssertionError("shifted reference without a transcript was accepted")
