"""
tests/test_ffmpeg_utils.py — Unit tests for safe FFmpeg subprocess utilities.
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from app.utils import ffmpeg_utils


def test_find_ffmpeg():
    # Should resolve successfully since FFmpeg is installed via Homebrew
    path = ffmpeg_utils.find_ffmpeg()
    assert Path(path).exists()
    assert "ffmpeg" in path.lower()


def test_find_ffprobe():
    path = ffmpeg_utils.find_ffprobe()
    assert Path(path).exists()
    assert "ffprobe" in path.lower()


@patch("subprocess.run")
def test_probe_video_parses_json(mock_run):
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout='{"streams": [{"codec_type": "video", "codec_name": "h264"}], "format": {"duration": "10.0"}}',
        stderr="",
    )
    result = ffmpeg_utils.probe_video(Path("/tmp/fake.mp4"))
    assert "streams" in result
    assert result["streams"][0]["codec_name"] == "h264"


@patch("app.utils.ffmpeg_utils._run")
def test_normalize_audio_enforces_canonical_audio_args(mock_run):
    in_p = Path("/tmp/dummy_in.mp4")
    out_p = Path("/tmp/dummy_out.mp4")
    ffmpeg_utils.normalize_audio(in_p, out_p)
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "-af" in cmd
    assert "loudnorm=I=-16:TP=-1.5:LRA=11" in cmd
    assert "-c:v" in cmd
    assert "copy" in cmd
    assert "-c:a" in cmd
    assert "aac" in cmd
    assert "-b:a" in cmd
    assert "192k" in cmd
    assert "-ar" in cmd
    assert "48000" in cmd
    assert "-ac" in cmd
    assert "2" in cmd


@patch("app.utils.ffmpeg_utils._run")
@patch("app.utils.ffmpeg_utils.probe_video")
def test_apply_audio_fade_enforces_canonical_audio_args(mock_probe, mock_run):
    mock_probe.return_value = {"format": {"duration": "10.0"}}
    in_p = Path("/tmp/dummy_in.mp4")
    out_p = Path("/tmp/dummy_out.mp4")
    ffmpeg_utils.apply_audio_fade(in_p, out_p, fade_in=0.3, fade_out=0.5)
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "-af" in cmd
    assert "afade=t=in:st=0:d=0.3,afade=t=out:st=9.500:d=0.5" in cmd
    assert "-c:v" in cmd
    assert "copy" in cmd
    assert "-c:a" in cmd
    assert "aac" in cmd
    assert "-b:a" in cmd
    assert "192k" in cmd
    assert "-ar" in cmd
    assert "48000" in cmd
    assert "-ac" in cmd
    assert "2" in cmd


def test_normalize_audio_preserves_canonical_48khz_stereo(tmp_path: Path):
    """Empirical test that loudnorm filter preserves 48,000 Hz stereo AAC output."""
    import subprocess
    src = tmp_path / "src_48k.mp4"
    ffmpeg = ffmpeg_utils.find_ffmpeg()
    subprocess.run([
        ffmpeg, "-y",
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=1",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
        "-c:v", "libx264", "-c:a", "aac", "-ar", "48000", "-ac", "2",
        str(src),
    ], check=True, capture_output=True)

    out = tmp_path / "normed.mp4"
    ffmpeg_utils.normalize_audio(src, out)
    assert out.exists()

    probe = ffmpeg_utils.probe_video(out)
    a = next(s for s in probe["streams"] if s["codec_type"] == "audio")
    assert int(a["sample_rate"]) == 48000
    assert int(a["channels"]) == 2
    assert a["codec_name"] == "aac"


def test_apply_audio_fade_preserves_canonical_48khz_stereo(tmp_path: Path):
    """Empirical test that apply_audio_fade preserves 48,000 Hz stereo AAC output."""
    import subprocess
    src = tmp_path / "src_48k.mp4"
    ffmpeg = ffmpeg_utils.find_ffmpeg()
    subprocess.run([
        ffmpeg, "-y",
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=1",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
        "-c:v", "libx264", "-c:a", "aac", "-ar", "48000", "-ac", "2",
        str(src),
    ], check=True, capture_output=True)

    out = tmp_path / "faded.mp4"
    ffmpeg_utils.apply_audio_fade(src, out, fade_in=0.1, fade_out=0.1)
    assert out.exists()

    probe = ffmpeg_utils.probe_video(out)
    a = next(s for s in probe["streams"] if s["codec_type"] == "audio")
    assert int(a["sample_rate"]) == 48000
    assert int(a["channels"]) == 2
    assert a["codec_name"] == "aac"
