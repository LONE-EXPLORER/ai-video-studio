"""
tests/test_silence_remover.py — Comprehensive unit and boundary tests for silence detection,
margin padding, and overlap merging (Milestone 2).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import pytest

from app.editing.silence_remover import (
    apply_margin_padding,
    detect_silence,
    get_speech_segments,
    merge_overlapping_segments,
)
from app.utils.ffmpeg_utils import find_ffmpeg, _run


# ---------------------------------------------------------------------------
# Synthetic Audio Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tone_with_silence_wav(tmp_path: Path) -> Tuple[Path, float]:
    """Generates a 3.0s WAV: 1.0s tone, 1.0s silence, 1.0s tone."""
    out = tmp_path / "tone_silence_tone.wav"
    ffmpeg = find_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-f", "lavfi", "-i", "sine=f=1000:d=1.0",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono:d=1.0",
        "-f", "lavfi", "-i", "sine=f=1000:d=1.0",
        "-filter_complex", "[0:a][1:a][2:a]concat=n=3:v=0:a=1[out]",
        "-map", "[out]",
        str(out),
    ]
    _run(cmd)
    return out, 3.0


@pytest.fixture
def all_silent_wav(tmp_path: Path) -> Tuple[Path, float]:
    """Generates a 2.0s completely silent WAV."""
    out = tmp_path / "all_silent.wav"
    ffmpeg = find_ffmpeg()
    cmd = [ffmpeg, "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono:d=2.0", str(out)]
    _run(cmd)
    return out, 2.0


@pytest.fixture
def continuous_tone_wav(tmp_path: Path) -> Tuple[Path, float]:
    """Generates a 2.0s continuous sine tone WAV."""
    out = tmp_path / "continuous_tone.wav"
    ffmpeg = find_ffmpeg()
    cmd = [ffmpeg, "-y", "-f", "lavfi", "-i", "sine=f=1000:d=2.0", str(out)]
    _run(cmd)
    return out, 2.0


# ---------------------------------------------------------------------------
# Unit Tests: Margin Padding & Clamping
# ---------------------------------------------------------------------------

def test_margin_padding_lead_and_tail():
    raw = [(1.000, 2.000)]
    padded = apply_margin_padding(raw, source_duration=5.0, lead_margin=0.080, tail_margin=0.180)
    assert padded == [(0.920, 2.180)]


def test_margin_padding_clamps_at_zero():
    raw = [(0.040, 1.000)]
    padded = apply_margin_padding(raw, source_duration=5.0, lead_margin=0.080, tail_margin=0.180)
    assert padded[0][0] == 0.000
    assert padded[0][1] == 1.180


def test_margin_padding_clamps_at_duration():
    raw = [(1.000, 4.950)]
    padded = apply_margin_padding(raw, source_duration=5.0, lead_margin=0.080, tail_margin=0.180)
    assert padded[0][0] == 0.920
    assert padded[0][1] == 5.000


def test_margin_padding_empty_input():
    assert apply_margin_padding([], source_duration=10.0) == []


def test_margin_padding_zero_duration():
    assert apply_margin_padding([(1.0, 2.0)], source_duration=0.0) == []
    assert apply_margin_padding([(1.0, 2.0)], source_duration=-1.0) == []


# ---------------------------------------------------------------------------
# Unit Tests: Overlap Merging
# ---------------------------------------------------------------------------

def test_overlap_merging_sub_margin_pause():
    # Gap is 0.10s. Lead=0.08, Tail=0.18 -> Total padding 0.26s -> Overlaps
    raw = [(0.500, 1.000), (1.100, 1.600)]
    padded = apply_margin_padding(raw, source_duration=5.0, merge=True)
    assert len(padded) == 1
    assert padded[0] == (0.420, 1.780)


def test_overlap_merging_cascade():
    # 3 segments overlapping consecutively
    raw = [(1.0, 2.0), (2.1, 3.0), (3.1, 4.0)]
    padded = apply_margin_padding(raw, source_duration=10.0, merge=True)
    assert len(padded) == 1
    assert padded[0] == (0.920, 4.180)


def test_overlap_merging_disjoint_segments():
    # Gap is 1.0s (> 0.26s) -> remains 2 separate segments
    raw = [(1.0, 2.0), (3.0, 4.0)]
    padded = apply_margin_padding(raw, source_duration=10.0, merge=True)
    assert len(padded) == 2
    assert padded[0] == (0.920, 2.180)
    assert padded[1] == (2.920, 4.180)


def test_overlap_merging_enclosed_interval():
    # A short segment inside a larger segment
    raw = [(1.0, 5.0), (2.0, 3.0)]
    merged = merge_overlapping_segments(raw)
    assert merged == [(1.0, 5.0)]


def test_overlap_merging_touching_intervals():
    raw = [(1.0, 2.0), (2.0, 3.0)]
    merged = merge_overlapping_segments(raw)
    assert merged == [(1.0, 3.0)]


def test_overlap_merging_unsorted_input():
    raw = [(3.0, 4.0), (1.0, 2.0)]
    merged = merge_overlapping_segments(raw)
    assert merged == [(1.0, 2.0), (3.0, 4.0)]


# ---------------------------------------------------------------------------
# Unit Tests: FFmpeg Stderr Parsing Hardening
# ---------------------------------------------------------------------------

@patch("subprocess.run")
def test_detect_silence_handles_unclosed_eof_silence(mock_run):
    mock_run.return_value = MagicMock(
        returncode=0,
        stderr=(
            "[silencedetect @ 0x1] silence_start: 2.500\n"
        ),
    )
    silences = detect_silence(Path("dummy.wav"), duration=5.0)
    assert silences == [(2.500, 5.000)]


@patch("subprocess.run")
def test_detect_silence_handles_leading_silence_without_start(mock_run):
    mock_run.return_value = MagicMock(
        returncode=0,
        stderr=(
            "[silencedetect @ 0x1] silence_end: 1.200 | silence_duration: 1.200\n"
            "[silencedetect @ 0x1] silence_start: 3.000\n"
            "[silencedetect @ 0x1] silence_end: 4.000 | silence_duration: 1.000\n"
        ),
    )
    silences = detect_silence(Path("dummy.wav"), duration=5.0)
    assert silences == [(0.000, 1.200), (3.000, 4.000)]


@patch("subprocess.run")
def test_detect_silence_handles_negative_timestamp(mock_run):
    mock_run.return_value = MagicMock(
        returncode=0,
        stderr=(
            "[silencedetect @ 0x1] silence_start: -0.005\n"
            "[silencedetect @ 0x1] silence_end: 1.000 | silence_duration: 1.005\n"
        ),
    )
    silences = detect_silence(Path("dummy.wav"), duration=5.0)
    assert silences == [(0.000, 1.000)]


# ---------------------------------------------------------------------------
# Integration Tests with FFmpeg
# ---------------------------------------------------------------------------

def test_detect_silence_real_synthetic_wav(tone_with_silence_wav):
    path, dur = tone_with_silence_wav
    silences = detect_silence(path, min_silence_duration=0.5, silence_threshold_db=-30.0, duration=dur)
    assert len(silences) == 1
    # Pause is between 1.0 and 2.0
    assert 0.9 <= silences[0][0] <= 1.1
    assert 1.9 <= silences[0][1] <= 2.1


def test_get_speech_segments_end_to_end(tone_with_silence_wav):
    path, dur = tone_with_silence_wav
    speech = get_speech_segments(
        path,
        duration=dur,
        min_silence_duration=0.5,
        silence_threshold_db=-30.0,
        lead_margin=0.080,
        tail_margin=0.180,
    )
    assert len(speech) == 2
    # First tone ~0.0 to 1.0 -> padded ~0.0 to 1.18
    assert speech[0][0] == 0.0
    assert 1.10 <= speech[0][1] <= 1.25
    # Second tone ~2.0 to 3.0 -> padded ~1.92 to 3.0
    assert 1.85 <= speech[1][0] <= 1.95
    assert speech[1][1] == 3.0


def test_get_speech_segments_all_silent(all_silent_wav):
    path, dur = all_silent_wav
    speech = get_speech_segments(path, duration=dur, min_silence_duration=0.5)
    assert speech == []


def test_get_speech_segments_continuous_tone(continuous_tone_wav):
    path, dur = continuous_tone_wav
    speech = get_speech_segments(path, duration=dur, min_silence_duration=0.5)
    assert speech == [(0.0, dur)]


def test_get_speech_segments_min_speech_duration(tone_with_silence_wav):
    path, dur = tone_with_silence_wav
    # Discard any speech segments shorter than 2.0s (both tones are 1.0s)
    speech = get_speech_segments(
        path,
        duration=dur,
        min_silence_duration=0.5,
        min_speech_duration=2.0,
    )
    assert speech == []
