"""
tests/conftest.py — Pytest fixtures for AI Video Studio test suite.
"""

from __future__ import annotations

import pytest
from pathlib import Path
import tempfile

from app.schemas.editing_plan import AspectRatio, CaptionSettings, Clip, EditingPlan
from app.schemas.transcript import Transcript, TranscriptSegment, TranscriptWord
from app.schemas.video_meta import AudioStream, VideoMetadata, VideoStream


@pytest.fixture
def tmp_jobs_dir(tmp_path: Path) -> Path:
    """Fixture providing a temporary jobs directory."""
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    return jobs_dir


@pytest.fixture
def sample_transcript() -> Transcript:
    """Fixture providing a realistic multi-segment transcript with word timestamps."""
    segments = [
        TranscriptSegment(
            id=1,
            start=0.5,
            end=4.0,
            text="Welcome to this video on AI video editing.",
            words=[
                TranscriptWord(word="Welcome", start=0.5, end=0.9),
                TranscriptWord(word="to", start=1.0, end=1.1),
                TranscriptWord(word="this", start=1.2, end=1.4),
                TranscriptWord(word="video", start=1.5, end=1.9),
                TranscriptWord(word="on", start=2.0, end=2.2),
                TranscriptWord(word="AI", start=2.3, end=2.6),
                TranscriptWord(word="video", start=2.7, end=3.1),
                TranscriptWord(word="editing.", start=3.2, end=4.0),
            ],
        ),
        TranscriptSegment(
            id=2,
            start=5.0,
            end=9.5,
            text="Today we build an entire studio right on Apple Silicon.",
            words=[
                TranscriptWord(word="Today", start=5.0, end=5.4),
                TranscriptWord(word="we", start=5.5, end=5.7),
                TranscriptWord(word="build", start=5.8, end=6.2),
                TranscriptWord(word="an", start=6.3, end=6.4),
                TranscriptWord(word="entire", start=6.5, end=7.0),
                TranscriptWord(word="studio", start=7.1, end=7.6),
                TranscriptWord(word="right", start=7.7, end=8.1),
                TranscriptWord(word="on", start=8.2, end=8.5),
                TranscriptWord(word="Apple", start=8.6, end=9.0),
                TranscriptWord(word="Silicon.", start=9.1, end=9.5),
            ],
        ),
    ]
    return Transcript(language="en", duration=10.0, segments=segments)


@pytest.fixture
def sample_editing_plan() -> EditingPlan:
    """Fixture providing a valid EditingPlan with two clips."""
    clips = [
        Clip(
            clip_id="c001",
            source_start=0.5,
            source_end=4.0,
            order=1,
            reason="Strong introductory hook",
        ),
        Clip(
            clip_id="c002",
            source_start=5.0,
            source_end=9.5,
            order=2,
            reason="Core value proposition",
        ),
    ]
    return EditingPlan(
        plan_id="test-plan-1",
        job_id="test-job-1",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=8.0,
        clips=clips,
        caption_settings=CaptionSettings(style="dynamic"),
        hook_summary="Welcome hook",
        edit_rationale="Fast-paced dynamic cut",
    )


@pytest.fixture
def sample_video_metadata() -> VideoMetadata:
    """Fixture providing a standard 1080p VideoMetadata object."""
    return VideoMetadata(
        file_path="/tmp/mock_video.mp4",
        file_size_mb=12.5,
        duration_seconds=30.0,
        video_stream=VideoStream(
            codec_name="h264",
            width=1920,
            height=1080,
            r_frame_rate="30/1",
            duration=30.0,
        ),
        audio_stream=AudioStream(
            codec_name="aac",
            sample_rate=48000,
            channels=2,
            duration=30.0,
        ),
        format_name="mov,mp4,m4a,3gp,3g2,mj2",
        has_audio=True,
        frame_rate=30.0,
    )
