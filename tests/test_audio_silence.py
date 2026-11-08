"""
tests/test_audio_silence.py — Unit tests for Phase 2 & 3 silence detection and audio processing.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from app.editing.silence_remover import detect_silence, get_speech_segments
from app.editing.audio_processor import duck_music


def test_detect_silence_on_synthetic_video():
    test_video = Path(__file__).resolve().parent.parent / "test_sample.mp4"
    if not test_video.exists():
        pytest.skip("test_sample.mp4 not generated yet")

    # Should run silencedetect without crashing
    silences = detect_silence(test_video, min_silence_duration=0.5, silence_threshold_db=-30.0)
    assert isinstance(silences, list)


def test_get_speech_segments():
    test_video = Path(__file__).resolve().parent.parent / "test_sample.mp4"
    if not test_video.exists():
        pytest.skip("test_sample.mp4 not generated yet")

    segments = get_speech_segments(test_video, duration=5.0, min_silence_duration=0.5)
    assert isinstance(segments, list)
