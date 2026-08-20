from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from search_reference_budget import (  # noqa: E402
    Boundary,
    enumerate_budget_windows,
    match_source_span,
    select_low_energy_boundaries,
)


def test_selects_separated_local_minima():
    values = [-10, -20, -45, -30, -50, -25, -15, -55, -20, -10]
    boundaries = select_low_energy_boundaries(values, 0.01, -40, 0.04)
    assert boundaries == [Boundary(0.07, -55.0)]


def test_enumerates_only_windows_inside_code_budget():
    boundaries = [
        Boundary(0.0, -50.0),
        Boundary(1.6, -40.0),
        Boundary(1.8, -60.0),
        Boundary(2.06, -45.0),
        Boundary(2.2, -70.0),
    ]
    windows = enumerate_budget_windows(boundaries, 80, 103)
    assert [window.predicted_codes for window in windows] == [80, 90, 103]


def test_keeps_acoustically_safer_window_for_same_budget():
    boundaries = [
        Boundary(0.0, -50.0),
        Boundary(0.2, -60.0),
        Boundary(1.8, -55.0),
        Boundary(2.0, -35.0),
    ]
    windows = enumerate_budget_windows(boundaries, 90, 90)
    assert len(windows) == 1
    assert windows[0].start_seconds == 0.0
    assert windows[0].end_seconds == 1.8


def test_matches_asr_text_to_exact_source_span():
    source = "What we need, helping us. You look at process changes."
    match = match_source_span(source, "you look at process changes.")
    assert match["reference_text"] == "You look at process changes."
    assert match["source_match_wer"] == 0.0
    assert match["source_span_ends_with_punctuation"] is True


def test_source_span_match_tolerates_one_asr_error():
    source = "The scope of work for a future procurement."
    match = match_source_span(source, "scope of work for a future project.")
    assert match["reference_text"] == "scope of work for a future procurement."
    assert 0.1 < match["source_match_wer"] < 0.2
