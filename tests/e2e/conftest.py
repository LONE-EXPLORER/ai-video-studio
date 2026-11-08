"""
tests/e2e/conftest.py — Comprehensive fixtures and synthetic media generators for E2E testing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

# Ensure standard Homebrew & bin paths in PATH
for p in ["/opt/homebrew/bin", "/usr/local/bin"]:
    if p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{p}:{os.environ.get('PATH', '')}"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.schemas.editing_plan import AspectRatio, CaptionSettings, Clip, EditingPlan, ZoomEffect, AudioSettings
from app.schemas.job import Job, JobStatus
from app.schemas.transcript import Transcript, TranscriptSegment, TranscriptWord
from app.schemas.video_meta import AudioStream, VideoMetadata, VideoStream
from app.utils.ffmpeg_utils import find_ffmpeg, find_ffprobe


# ---------------------------------------------------------------------------
# Synthetic Media Generation Utilities
# ---------------------------------------------------------------------------

def make_synthetic_video(
    output_path: Path,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
    duration: float = 1.0,
    pattern: str = "testsrc",
    has_audio: bool = True,
    audio_freq: int = 1000,
    sample_rate: int = 48000,
) -> Path:
    """
    Generate a valid synthetic MP4 video file using FFmpeg lavfi filters.
    Runs in < 200ms.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()

    cmd = [ffmpeg, "-y"]

    # Video input
    if pattern == "color":
        cmd += ["-f", "lavfi", "-i", f"color=c=blue:s={width}x{height}:r={fps}:d={duration}"]
    elif pattern == "smptebars":
        cmd += ["-f", "lavfi", "-i", f"smptebars=size={width}x{height}:rate={fps}:duration={duration}"]
    elif pattern == "testsrc2":
        cmd += ["-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate={fps}:duration={duration}"]
    else:
        cmd += ["-f", "lavfi", "-i", f"testsrc=size={width}x{height}:rate={fps}:duration={duration}"]

    # Audio input
    if has_audio:
        cmd += [
            "-f", "lavfi",
            "-i", f"sine=frequency={audio_freq}:duration={duration}:sample_rate={sample_rate}",
            "-c:a", "aac",
            "-b:a", "128k",
            "-ar", str(sample_rate),
            "-ac", "2",
        ]
    else:
        cmd += ["-an"]

    cmd += [
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-t", str(duration),
        str(output_path),
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg synthetic video failed: {res.stderr}")

    return output_path


def make_synthetic_audio(
    output_path: Path,
    duration: float = 1.0,
    sample_rate: int = 48000,
    channels: int = 2,
    freq: int = 440,
    audio_format: str = "wav",
) -> Path:
    """
    Generate a synthetic clean audio file (WAV or AAC) using FFmpeg.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()

    cmd = [
        ffmpeg, "-y",
        "-f", "lavfi",
        "-i", f"sine=frequency={freq}:duration={duration}:sample_rate={sample_rate}",
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-t", str(duration),
        str(output_path),
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg synthetic audio failed: {res.stderr}")

    return output_path


def probe_media_file(path: Path) -> Dict[str, Any]:
    """
    Probe a video/audio file with ffprobe and return stream metadata.
    """
    ffprobe = find_ffprobe()
    cmd = [
        ffprobe,
        "-v", "error",
        "-show_format",
        "-show_streams",
        "-of", "json",
        str(path),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFprobe failed on {path}: {res.stderr}")
    return json.loads(res.stdout)


# ---------------------------------------------------------------------------
# Session & Module Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def session_temp_media_dir(tmp_path_factory) -> Path:
    """Shared temporary directory for cached session-wide synthetic media."""
    d = tmp_path_factory.mktemp("e2e_synthetic_media")
    return d


@pytest.fixture(scope="session")
def synth_video_vertical(session_temp_media_dir: Path) -> Path:
    """1080x1920 30fps vertical video, 2.0s with 48kHz audio."""
    out = session_temp_media_dir / "synth_vertical_1080x1920.mp4"
    if not out.exists():
        make_synthetic_video(out, width=1080, height=1920, fps=30, duration=2.0, has_audio=True)
    return out


@pytest.fixture(scope="session")
def synth_video_horizontal(session_temp_media_dir: Path) -> Path:
    """1920x1080 30fps landscape video, 2.0s with 48kHz audio."""
    out = session_temp_media_dir / "synth_horizontal_1920x1080.mp4"
    if not out.exists():
        make_synthetic_video(out, width=1920, height=1080, fps=30, duration=2.0, has_audio=True)
    return out


@pytest.fixture(scope="session")
def synth_video_square(session_temp_media_dir: Path) -> Path:
    """1080x1080 24fps square video, 2.0s with audio."""
    out = session_temp_media_dir / "synth_square_1080x1080.mp4"
    if not out.exists():
        make_synthetic_video(out, width=1080, height=1080, fps=24, duration=2.0, has_audio=True)
    return out


@pytest.fixture(scope="session")
def synth_video_silent(session_temp_media_dir: Path) -> Path:
    """1080x1920 30fps vertical video without audio track."""
    out = session_temp_media_dir / "synth_silent.mp4"
    if not out.exists():
        make_synthetic_video(out, width=1080, height=1920, fps=30, duration=1.5, has_audio=False)
    return out


@pytest.fixture(scope="session")
def synth_broll_clip(session_temp_media_dir: Path) -> Path:
    """1080x1920 B-roll clip, 1.5s color testsrc."""
    out = session_temp_media_dir / "synth_broll.mp4"
    if not out.exists():
        make_synthetic_video(out, width=1080, height=1920, fps=30, duration=1.5, pattern="color")
    return out


@pytest.fixture(scope="session")
def synth_sfx_whoosh(session_temp_media_dir: Path) -> Path:
    """Synthetic whoosh SFX (0.3s WAV)."""
    out = session_temp_media_dir / "whoosh.wav"
    if not out.exists():
        make_synthetic_audio(out, duration=0.3, freq=600)
    return out


@pytest.fixture(scope="session")
def synth_sfx_pop(session_temp_media_dir: Path) -> Path:
    """Synthetic pop SFX (0.15s WAV)."""
    out = session_temp_media_dir / "pop.wav"
    if not out.exists():
        make_synthetic_audio(out, duration=0.15, freq=1200)
    return out


@pytest.fixture(scope="session")
def synth_bgm_track(session_temp_media_dir: Path) -> Path:
    """Synthetic background music track (4.0s WAV)."""
    out = session_temp_media_dir / "background_music.wav"
    if not out.exists():
        make_synthetic_audio(out, duration=4.0, freq=300)
    return out


@pytest.fixture
def isolated_job_workspace(tmp_path: Path) -> Dict[str, Path]:
    """Isolated directory structure mimicking standard job asset directories."""
    job_id = "e2e_test_job"
    root = tmp_path / job_id
    dirs = {
        "root": root,
        "input": root / "input",
        "intermediate": root / "intermediate",
        "output": root / "output",
        "audio": root / "audio",
        "frames": root / "frames",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


@pytest.fixture
def trump_asset_video() -> Optional[Path]:
    """Path to the primary test video trump_5_day_trap.mp4 if present."""
    path = _PROJECT_ROOT / "assets" / "test_videos" / "trump_5_day_trap.mp4"
    if path.exists() and path.stat().st_size > 0:
        return path
    return None


@pytest.fixture
def sample_e2e_transcript() -> Transcript:
    """Multi-word realistic transcript for speech and caption testing."""
    return Transcript(
        language="en",
        duration=3.0,
        segments=[
            TranscriptSegment(
                id=1,
                start=0.2,
                end=2.8,
                text="Transform your local video editing with Apple Silicon",
                words=[
                    TranscriptWord(word="Transform", start=0.2, end=0.6),
                    TranscriptWord(word="your", start=0.7, end=0.9),
                    TranscriptWord(word="local", start=1.0, end=1.3),
                    TranscriptWord(word="video", start=1.4, end=1.7),
                    TranscriptWord(word="editing", start=1.8, end=2.2),
                    TranscriptWord(word="with", start=2.3, end=2.4),
                    TranscriptWord(word="Apple", start=2.5, end=2.6),
                    TranscriptWord(word="Silicon", start=2.65, end=2.8),
                ],
            )
        ],
    )
