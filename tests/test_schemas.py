"""
tests/test_schemas.py — Unit tests for Pydantic data schemas.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.editing_plan import AspectRatio, Clip, EditingPlan
from app.schemas.transcript import Transcript, TranscriptSegment, TranscriptWord
from app.schemas.video_meta import VideoMetadata


def test_clip_duration_calculation():
    clip = Clip(
        clip_id="c1",
        source_start=2.0,
        source_end=7.5,
        order=1,
        reason="test clip",
    )
    assert clip.duration == 5.5


def test_clip_invalid_timestamps():
    with pytest.raises(ValidationError):
        # End before start
        Clip(
            clip_id="c2",
            source_start=10.0,
            source_end=5.0,
            order=1,
            reason="invalid",
        )


def test_editing_plan_total_duration(sample_editing_plan: EditingPlan):
    assert sample_editing_plan.total_duration == (3.5 + 4.5)


def test_transcript_text_and_words_in_range(sample_transcript: Transcript):
    assert "Welcome to this video" in sample_transcript.full_text()
    words = sample_transcript.words_in_range(0.0, 3.0)
    assert len(words) > 0
    assert any(w.word == "Welcome" for w in words)


def test_transcript_srt_export(sample_transcript: Transcript):
    srt = sample_transcript.to_srt()
    assert "-->" in srt
    assert "Welcome to this video" in srt
