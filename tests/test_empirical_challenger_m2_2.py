"""
tests/test_empirical_challenger_m2_2.py — Empirical Challenger Stress Test Suite for Milestone 2.

Empirically verifies:
1. Discrete 1.15x jump punch-in video quality (1080x1920 30fps VideoToolbox & software fallback).
2. Static magnification across all frames (instantaneous jump interrupt, not Ken Burns creep).
3. Sample Aspect Ratio (SAR) preservation on punch-in crop.
4. J-cut audio lead delivery in final muxed video container.
5. L-cut audio tail delivery in final muxed video container.
6. Container duration consistency (sum of video clip durations).
7. Extreme boundary clamping for split edits (lead/tail > clip durations).
8. Micro audio fade boundary envelope continuity.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest
from PIL import Image

from app.editing.editor import VideoEditor
from app.schemas.editing_plan import AspectRatio, AudioSettings, Clip, EditingPlan, ZoomEffect
from app.utils import ffmpeg_utils


def _generate_synthetic_tone_video(
    output_path: Path,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
    duration: float = 5.0,
    frequency: int = 440,
    sample_rate: int = 48000,
) -> Path:
    """Generate synthetic video with specified sine audio tone."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_utils.find_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-f", "lavfi", "-i", f"testsrc=size={width}x{height}:rate={fps}:duration={duration}",
        "-f", "lavfi", "-i", f"sine=frequency={frequency}:sample_rate={sample_rate}:duration={duration}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", str(sample_rate), "-ac", "2",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def _extract_pcm_spectrum(
    media_path: Path,
    start_sec: float,
    end_sec: float,
    target_sample_rate: int = 48000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract audio window as PCM and return (frequencies, fft_magnitudes)."""
    ffmpeg = ffmpeg_utils.find_ffmpeg()
    wav_path = media_path.parent / f"extract_{start_sec:.2f}_{end_sec:.2f}.wav"
    cmd = [
        ffmpeg, "-y",
        "-ss", str(start_sec),
        "-to", str(end_sec),
        "-i", str(media_path),
        "-vn",
        "-c:a", "pcm_s16le",
        "-ar", str(target_sample_rate),
        "-ac", "1",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    with wave.open(str(wav_path), "rb") as wf:
        nframes = wf.getnframes()
        frames = wf.readframes(nframes)
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)

    fft_mag = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), 1.0 / target_sample_rate)
    return freqs, fft_mag


# ===========================================================================
# 1. Discrete 1.15x Jump Punch-In Video Tests
# ===========================================================================

def test_punch_in_videotoolbox_properties(tmp_path: Path):
    """Verify 1.15x punch-in produces 1080x1920 30fps with 48kHz stereo AAC via VideoToolbox."""
    src = _generate_synthetic_tone_video(tmp_path / "src_1080.mp4", width=1080, height=1920, fps=30, duration=3.0)
    zoomed = tmp_path / "zoomed_vt.mp4"

    ffmpeg_utils.apply_zoom_effect(
        input_path=src,
        output_path=zoomed,
        zoom_factor=1.15,
        discrete=True,
        use_videotoolbox=True,
    )

    probe = ffmpeg_utils.probe_video(zoomed)
    vstream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    astream = next(s for s in probe["streams"] if s["codec_type"] == "audio")

    assert vstream["width"] == 1080
    assert vstream["height"] == 1920
    assert vstream["r_frame_rate"] == "30/1"
    assert astream["sample_rate"] == "48000"
    assert astream["channels"] == 2
    assert float(probe["format"]["duration"]) == pytest.approx(3.0, abs=0.1)


def test_punch_in_static_magnification_vs_ken_burns(tmp_path: Path):
    """Verify discrete punch-in maintains static magnification across all frames."""
    ffmpeg = ffmpeg_utils.find_ffmpeg()
    src = tmp_path / "static_bars.mp4"
    cmd = [
        ffmpeg, "-y",
        "-f", "lavfi", "-i", "smptebars=size=1080x1920:rate=30:duration=3.0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(src),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    # 1. Discrete zoom
    zoomed_discrete = tmp_path / "zoomed_discrete.mp4"
    ffmpeg_utils.apply_zoom_effect(
        input_path=src,
        output_path=zoomed_discrete,
        zoom_factor=1.15,
        discrete=True,
    )

    # 2. Continuous slow zoom
    zoomed_continuous = tmp_path / "zoomed_continuous.mp4"
    ffmpeg_utils.apply_zoom_effect(
        input_path=src,
        output_path=zoomed_continuous,
        zoom_factor=1.15,
        discrete=False,
    )

    f0_d = tmp_path / "f0_d.png"
    f1_d = tmp_path / "f1_d.png"
    subprocess.run([ffmpeg, "-y", "-ss", "0.2", "-i", str(zoomed_discrete), "-frames:v", "1", str(f0_d)], check=True, capture_output=True)
    subprocess.run([ffmpeg, "-y", "-ss", "2.0", "-i", str(zoomed_discrete), "-frames:v", "1", str(f1_d)], check=True, capture_output=True)

    f0_c = tmp_path / "f0_c.png"
    f1_c = tmp_path / "f1_c.png"
    subprocess.run([ffmpeg, "-y", "-ss", "0.2", "-i", str(zoomed_continuous), "-frames:v", "1", str(f0_c)], check=True, capture_output=True)
    subprocess.run([ffmpeg, "-y", "-ss", "2.0", "-i", str(zoomed_continuous), "-frames:v", "1", str(f1_c)], check=True, capture_output=True)

    img0_d = np.array(Image.open(f0_d), dtype=float)
    img1_d = np.array(Image.open(f1_d), dtype=float)
    mae_discrete = np.mean(np.abs(img0_d - img1_d))

    img0_c = np.array(Image.open(f0_c), dtype=float)
    img1_c = np.array(Image.open(f1_c), dtype=float)
    mae_continuous = np.mean(np.abs(img0_c - img1_c))

    # Discrete zoom must be static (MAE near zero), unlike continuous Ken Burns zoom
    assert mae_discrete < 1.0, f"Discrete zoom frames diverged: MAE={mae_discrete}"
    assert mae_continuous > 5.0, f"Continuous zoom did not ramp: MAE={mae_continuous}"


def test_punch_in_software_fallback(tmp_path: Path):
    """Verify apply_zoom_effect software fallback (libx264) works seamlessly."""
    src = _generate_synthetic_tone_video(tmp_path / "src_sw.mp4", width=1080, height=1920, duration=2.0)
    zoomed_sw = tmp_path / "zoomed_sw.mp4"

    ffmpeg_utils.apply_zoom_effect(
        input_path=src,
        output_path=zoomed_sw,
        zoom_factor=1.15,
        discrete=True,
        use_videotoolbox=False,
    )

    probe = ffmpeg_utils.probe_video(zoomed_sw)
    vstream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert vstream["width"] == 1080
    assert vstream["height"] == 1920


def test_punch_in_sample_aspect_ratio_sar(tmp_path: Path):
    """Stress-test: Check if 1.15x zoom crop maintains 1:1 square pixel aspect ratio."""
    src = _generate_synthetic_tone_video(tmp_path / "src_sar.mp4", width=1080, height=1920, duration=1.0)
    zoomed = tmp_path / "zoomed_sar.mp4"

    ffmpeg_utils.apply_zoom_effect(
        input_path=src,
        output_path=zoomed,
        zoom_factor=1.15,
        discrete=True,
    )

    probe = ffmpeg_utils.probe_video(zoomed)
    vstream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    sar = vstream.get("sample_aspect_ratio", "1:1")
    # If SAR is not 1:1 (e.g. 3752:3753 due to rounding in 938x1668 crop), note discrepancy
    assert sar in ("1:1", None), f"Sample aspect ratio is distorted: {sar}"


# ===========================================================================
# 2. Split Edit (J-Cut / L-Cut) Multi-Track Assembly Tests
# ===========================================================================

def test_jcut_audio_lead_present_in_muxed_output(tmp_path: Path):
    """
    CRITICAL EMPIRICAL TEST:
    Verify that in a J-cut (audio_lead = 0.5s), the incoming clip's audio (1000Hz)
    is actually mixed and present in the final output container during the preceding
    clip's video interval (1.6s - 1.9s).
    """
    inter_dir = tmp_path / "inter"
    inter_dir.mkdir(parents=True, exist_ok=True)

    src1 = _generate_synthetic_tone_video(tmp_path / "src1.mp4", width=320, height=240, duration=5.0, frequency=440)
    src2 = _generate_synthetic_tone_video(tmp_path / "src2.mp4", width=320, height=240, duration=5.0, frequency=1000)

    plan = EditingPlan(
        plan_id="p_jcut",
        job_id="j_jcut",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=4.0,
        hook_summary="hook",
        edit_rationale="jcut test",
        clips=[
            Clip(clip_id="c1", source_id="s1", source_start=0.0, source_end=2.0, order=0, reason="c1"),
            Clip(clip_id="c2", source_id="s2", source_start=1.0, source_end=3.0, order=1, reason="c2", audio_lead=0.5),
        ],
        audio_settings=AudioSettings(normalize=False, fade_in=0.0, fade_out=0.0),
    )

    editor = VideoEditor()
    out = editor.execute_plan(
        plan=plan,
        source_path={"s1": src1, "s2": src2},
        job_dirs={"intermediate": inter_dir},
    )

    # Analyze audio in J-cut transition window (1.6s to 1.9s)
    freqs, fft_mag = _extract_pcm_spectrum(out, start_sec=1.6, end_sec=1.9)
    power_440 = fft_mag[np.argmin(np.abs(freqs - 440))]
    power_1000 = fft_mag[np.argmin(np.abs(freqs - 1000))]

    # 1000Hz tone from Clip 2 MUST be present in the J-cut lead window
    assert power_1000 > 10000.0, (
        f"J-cut audio lead missing from final output container! "
        f"Power at 1000Hz={power_1000:.1f}, Power at 440Hz={power_440:.1f}. "
        f"Mux command failed to map rough_audio_mixed into rough_cut_raw."
    )


def test_lcut_audio_tail_present_in_muxed_output(tmp_path: Path):
    """
    CRITICAL EMPIRICAL TEST:
    Verify that in an L-cut (audio_tail = 0.5s), the outgoing clip's audio (440Hz)
    is actually mixed and present in the final output container during the succeeding
    clip's video interval (2.1s - 2.4s).
    """
    inter_dir = tmp_path / "inter"
    inter_dir.mkdir(parents=True, exist_ok=True)

    src1 = _generate_synthetic_tone_video(tmp_path / "src1.mp4", width=320, height=240, duration=5.0, frequency=440)
    src2 = _generate_synthetic_tone_video(tmp_path / "src2.mp4", width=320, height=240, duration=5.0, frequency=1000)

    plan = EditingPlan(
        plan_id="p_lcut",
        job_id="j_lcut",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=4.0,
        hook_summary="hook",
        edit_rationale="lcut test",
        clips=[
            Clip(clip_id="c1", source_id="s1", source_start=0.0, source_end=2.0, order=0, reason="c1", audio_tail=0.5),
            Clip(clip_id="c2", source_id="s2", source_start=0.0, source_end=2.0, order=1, reason="c2"),
        ],
        audio_settings=AudioSettings(normalize=False, fade_in=0.0, fade_out=0.0),
    )

    editor = VideoEditor()
    out = editor.execute_plan(
        plan=plan,
        source_path={"s1": src1, "s2": src2},
        job_dirs={"intermediate": inter_dir},
    )

    # Analyze audio in L-cut transition window (2.1s to 2.4s)
    freqs, fft_mag = _extract_pcm_spectrum(out, start_sec=2.1, end_sec=2.4)
    power_440 = fft_mag[np.argmin(np.abs(freqs - 440))]
    power_1000 = fft_mag[np.argmin(np.abs(freqs - 1000))]

    # 440Hz tone from Clip 1 MUST be present in the L-cut tail window
    assert power_440 > 10000.0, (
        f"L-cut audio tail missing from final output container! "
        f"Power at 440Hz={power_440:.1f}, Power at 1000Hz={power_1000:.1f}. "
        f"Mux command failed to map rough_audio_mixed into rough_cut_raw."
    )


def test_split_edit_container_duration_matches_master_video(tmp_path: Path):
    """Verify net output container duration equals sum of video clip durations."""
    inter_dir = tmp_path / "inter"
    inter_dir.mkdir(parents=True, exist_ok=True)

    src1 = _generate_synthetic_tone_video(tmp_path / "src1.mp4", duration=5.0)
    src2 = _generate_synthetic_tone_video(tmp_path / "src2.mp4", duration=5.0)

    plan = EditingPlan(
        plan_id="p_dur",
        job_id="j_dur",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=4.0,
        hook_summary="hook",
        edit_rationale="duration test",
        clips=[
            Clip(clip_id="c1", source_id="s1", source_start=0.0, source_end=2.0, order=0, reason="c1", audio_tail=0.3),
            Clip(clip_id="c2", source_id="s2", source_start=1.0, source_end=3.0, order=1, reason="c2", audio_lead=0.3),
        ],
        audio_settings=AudioSettings(normalize=False, fade_in=0.0, fade_out=0.0),
    )

    editor = VideoEditor()
    out = editor.execute_plan(
        plan=plan,
        source_path={"s1": src1, "s2": src2},
        job_dirs={"intermediate": inter_dir},
    )

    probe = ffmpeg_utils.probe_video(out)
    total_dur = float(probe["format"]["duration"])
    # Video duration = 2.0 + 2.0 = 4.0s
    assert total_dur == pytest.approx(4.0, abs=0.15)


def test_split_edit_boundary_clamping_safety(tmp_path: Path):
    """Verify excessive audio_lead and audio_tail are clamped to safe clip boundaries without error."""
    inter_dir = tmp_path / "inter"
    inter_dir.mkdir(parents=True, exist_ok=True)

    src1 = _generate_synthetic_tone_video(tmp_path / "src1.mp4", duration=4.0)
    src2 = _generate_synthetic_tone_video(tmp_path / "src2.mp4", duration=4.0)

    # Set audio_lead = 50.0 (exceeds preceding clip) and audio_tail = 50.0 (exceeds following clip)
    plan = EditingPlan(
        plan_id="p_clamp",
        job_id="j_clamp",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=4.0,
        hook_summary="hook",
        edit_rationale="clamp test",
        clips=[
            Clip(clip_id="c1", source_id="s1", source_start=0.0, source_end=2.0, order=0, reason="c1", audio_tail=50.0),
            Clip(clip_id="c2", source_id="s2", source_start=1.0, source_end=3.0, order=1, reason="c2", audio_lead=50.0),
        ],
        audio_settings=AudioSettings(normalize=False, fade_in=0.0, fade_out=0.0),
    )

    editor = VideoEditor()
    out = editor.execute_plan(
        plan=plan,
        source_path={"s1": src1, "s2": src2},
        job_dirs={"intermediate": inter_dir},
    )

    assert out.exists()
    assert out.stat().st_size > 0
