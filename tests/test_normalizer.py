"""
tests/test_normalizer.py — Unit tests for MediaNormalizer.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
import pytest

from app.editing.normalizer import MediaNormalizer
from app.utils import ffmpeg_utils


def _create_synthetic_clip(
    path: Path,
    width: int = 640,
    height: int = 360,
    fps: int = 25,
    duration: float = 1.0,
    with_audio: bool = True,
    sample_rate: int = 44100,
) -> Path:
    """Helper to generate a lightweight test video with lavfi."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_utils.find_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-f", "lavfi",
        "-i", f"testsrc=size={width}x{height}:rate={fps}:duration={duration}",
    ]
    if with_audio:
        cmd.extend([
            "-f", "lavfi",
            "-i", f"sine=frequency=440:sample_rate={sample_rate}:duration={duration}",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-ar", str(sample_rate),
            "-ac", "2",
        ])
    else:
        cmd.extend([
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
        ])
    cmd.append(str(path))
    subprocess.run(cmd, check=True, capture_output=True)
    return path


def test_normalizer_initialization_defaults():
    norm = MediaNormalizer()
    assert norm.target_w == 1080
    assert norm.target_h == 1920
    assert norm.fps == 30
    assert norm.video_bitrate == "6000k"
    assert norm.audio_bitrate == "192k"


def test_normalizer_initialization_custom():
    norm = MediaNormalizer(target_w=720, target_h=1280, fps=24, video_bitrate="3000k", audio_bitrate="128k")
    assert norm.target_w == 720
    assert norm.target_h == 1280
    assert norm.fps == 24
    assert norm.video_bitrate == "3000k"
    assert norm.audio_bitrate == "128k"


def test_normalizer_nonexistent_file_raises_filenotfound(tmp_path: Path):
    norm = MediaNormalizer()
    nonexistent = tmp_path / "does_not_exist.mp4"
    out = tmp_path / "out.mp4"
    with pytest.raises(FileNotFoundError, match="Source video file not found"):
        norm.conform_clip(nonexistent, out)


def test_normalizer_conform_clip_letterbox(tmp_path: Path):
    src = tmp_path / "raw_landscape.mp4"
    _create_synthetic_clip(src, width=640, height=360, fps=25, duration=1.0, with_audio=True)

    out = tmp_path / "conformed_letterbox.mp4"
    norm = MediaNormalizer(target_w=1080, target_h=1920, fps=30)
    result = norm.conform_clip(src, out, pad_mode="letterbox")

    assert result.exists()
    assert result == out
    info = ffmpeg_utils.probe_video(result)
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
    astream = next(s for s in info["streams"] if s["codec_type"] == "audio")

    assert int(vstream["width"]) == 1080
    assert int(vstream["height"]) == 1920
    assert int(astream["sample_rate"]) == 48000
    assert int(astream["channels"]) == 2
    assert astream["codec_name"] == "aac"


def test_normalizer_conform_clip_blur_mode(tmp_path: Path):
    src = tmp_path / "raw_landscape_blur.mp4"
    _create_synthetic_clip(src, width=640, height=360, fps=25, duration=1.0, with_audio=True)

    out = tmp_path / "conformed_blur.mp4"
    norm = MediaNormalizer(target_w=1080, target_h=1920, fps=30)
    result = norm.conform_clip(src, out, pad_mode="blur")

    assert result.exists()
    info = ffmpeg_utils.probe_video(result)
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert int(vstream["width"]) == 1080
    assert int(vstream["height"]) == 1920


def test_normalizer_conform_clip_missing_audio_synthetic_silence(tmp_path: Path):
    src = tmp_path / "silent_raw.mp4"
    _create_synthetic_clip(src, width=320, height=240, fps=25, duration=1.0, with_audio=False)

    # Confirm source has no audio
    src_probe = ffmpeg_utils.probe_video(src)
    assert not any(s["codec_type"] == "audio" for s in src_probe["streams"])

    out = tmp_path / "conformed_silent.mp4"
    norm = MediaNormalizer(target_w=1080, target_h=1920, fps=30)
    result = norm.conform_clip(src, out, pad_mode="letterbox")

    assert result.exists()
    info = ffmpeg_utils.probe_video(result)
    astream = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    assert astream is not None, "Conformed video must have an audio track injected"
    assert int(astream["sample_rate"]) == 48000
    assert int(astream["channels"]) == 2


def test_normalizer_conform_sources_multi(tmp_path: Path):
    src1 = tmp_path / "cam_a.mp4"
    src2 = tmp_path / "cam_b.mp4"
    _create_synthetic_clip(src1, width=640, height=360, fps=25, duration=0.8, with_audio=True)
    _create_synthetic_clip(src2, width=360, height=640, fps=30, duration=0.8, with_audio=True)

    out_dir = tmp_path / "conformed_dir"
    norm = MediaNormalizer(target_w=1080, target_h=1920, fps=30)
    source_map = norm.conform_sources([src1, src2], out_dir)

    assert "source_0" in source_map
    assert "source_1" in source_map
    assert source_map["source_0"].exists()
    assert source_map["source_1"].exists()

    meta0 = ffmpeg_utils.probe_video(source_map["source_0"])
    v0 = next(s for s in meta0["streams"] if s["codec_type"] == "video")
    assert int(v0["width"]) == 1080
    assert int(v0["height"]) == 1920

    meta1 = ffmpeg_utils.probe_video(source_map["source_1"])
    v1 = next(s for s in meta1["streams"] if s["codec_type"] == "video")
    assert int(v1["width"]) == 1080
    assert int(v1["height"]) == 1920
