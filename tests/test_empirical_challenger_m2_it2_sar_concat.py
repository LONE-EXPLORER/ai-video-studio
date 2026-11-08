"""
tests/test_empirical_challenger_m2_it2_sar_concat.py — Empirical Challenger Stress Test Suite
for Milestone 2 Iteration 2.

Focus Areas:
1. Discrete 1.15x punch-in zoom Sample Aspect Ratio (SAR) strictly 1:1 across zoom factors.
2. Heterogeneous alternating clip concatenation stability (1.0x wide and 1.15x zoom).
3. Concat via FFmpeg filter_complex concat ensuring exit code 0 (preventing exit code 234 SAR mismatch).
4. Full probe verification of SAR 1:1, DAR 9:16, 1080x1920, 30fps.
5. End-to-end VideoEditor integration combining discrete punch-ins and J/L split-edits.
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


def _create_synthetic_test_video(
    output_path: Path,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
    duration: float = 2.0,
    frequency: int = 440,
    sample_rate: int = 48000,
) -> Path:
    """Generate a clean synthetic 1080x1920 30fps test video with audio tone."""
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
    """Extract audio PCM slice and compute FFT magnitudes."""
    ffmpeg = ffmpeg_utils.find_ffmpeg()
    wav_path = media_path.parent / f"fft_{start_sec:.2f}_{end_sec:.2f}.wav"
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
# 1. SAR and DAR Consistency Across Zoom Factors (VideoToolbox & Software)
# ===========================================================================

@pytest.mark.parametrize("zoom_factor", [1.05, 1.10, 1.15, 1.20, 1.25, 1.30])
@pytest.mark.parametrize("use_videotoolbox", [True, False])
def test_empirical_sar_across_zoom_factors(tmp_path: Path, zoom_factor: float, use_videotoolbox: bool):
    """
    Empirical Test: Verify that apply_zoom_effect maintains strict 1:1 SAR and 9:16 DAR
    across various zoom factors under both VideoToolbox hardware encoding and software fallback.
    """
    src = _create_synthetic_test_video(tmp_path / f"src_{zoom_factor}_{use_videotoolbox}.mp4", duration=1.0)
    zoomed = tmp_path / f"zoomed_{zoom_factor}_{use_videotoolbox}.mp4"

    ffmpeg_utils.apply_zoom_effect(
        input_path=src,
        output_path=zoomed,
        zoom_factor=zoom_factor,
        discrete=True,
        use_videotoolbox=use_videotoolbox,
    )

    probe = ffmpeg_utils.probe_video(zoomed)
    vstream = next(s for s in probe["streams"] if s["codec_type"] == "video")

    sar = vstream.get("sample_aspect_ratio", "1:1")
    dar = vstream.get("display_aspect_ratio", "9:16")

    assert sar in ("1:1", None), f"Distorted SAR for zoom={zoom_factor}, vt={use_videotoolbox}: {sar}"
    assert dar in ("9:16", None), f"Distorted DAR for zoom={zoom_factor}, vt={use_videotoolbox}: {dar}"
    assert vstream["width"] == 1080
    assert vstream["height"] == 1920
    assert vstream["r_frame_rate"] == "30/1"


# ===========================================================================
# 2. Heterogeneous Alternating Clip Concatenation (Filter Complex Concat)
# ===========================================================================

def test_heterogeneous_alternating_concat_stability(tmp_path: Path):
    """
    Empirical Stress Test:
    Assemble 6 heterogeneous clips alternating between wide (1.0x) and discrete punch-in zooms
    (1.15x VideoToolbox, 1.15x libx264, 1.20x VideoToolbox, 1.10x libx264).
    Concatenate with filter_complex concat and verify exit code 0 (no exit code 234 mismatch).
    """
    ffmpeg = ffmpeg_utils.find_ffmpeg()

    # Generate 6 distinct clips with distinct frequencies
    clip0_wide = _create_synthetic_test_video(tmp_path / "c0_wide.mp4", duration=1.2, frequency=440)
    clip1_vt_115 = tmp_path / "c1_vt_115.mp4"
    clip2_wide = _create_synthetic_test_video(tmp_path / "c2_wide.mp4", duration=1.2, frequency=660)
    clip3_sw_115 = tmp_path / "c3_sw_115.mp4"
    clip4_vt_120 = tmp_path / "c4_vt_120.mp4"
    clip5_wide = _create_synthetic_test_video(tmp_path / "c5_wide.mp4", duration=1.2, frequency=880)

    # Produce zoomed clips
    c1_raw = _create_synthetic_test_video(tmp_path / "c1_raw.mp4", duration=1.2, frequency=550)
    ffmpeg_utils.apply_zoom_effect(c1_raw, clip1_vt_115, zoom_factor=1.15, discrete=True, use_videotoolbox=True)

    c3_raw = _create_synthetic_test_video(tmp_path / "c3_raw.mp4", duration=1.2, frequency=770)
    ffmpeg_utils.apply_zoom_effect(c3_raw, clip3_sw_115, zoom_factor=1.15, discrete=True, use_videotoolbox=False)

    c4_raw = _create_synthetic_test_video(tmp_path / "c4_raw.mp4", duration=1.2, frequency=990)
    ffmpeg_utils.apply_zoom_effect(c4_raw, clip4_vt_120, zoom_factor=1.20, discrete=True, use_videotoolbox=True)

    clips = [clip0_wide, clip1_vt_115, clip2_wide, clip3_sw_115, clip4_vt_120, clip5_wide]

    # Verify SAR on every individual clip before concatenation
    for i, c in enumerate(clips):
        p = ffmpeg_utils.probe_video(c)
        vs = next(s for s in p["streams"] if s["codec_type"] == "video")
        sar = vs.get("sample_aspect_ratio", "1:1")
        assert sar in ("1:1", None), f"Clip {i} ({c.name}) has non-square SAR: {sar}"

    # 1. Concatenate using ffmpeg_utils.concat_clips (which uses filter_complex concat Strategy A)
    concat_out = tmp_path / "concatenated_output.mp4"
    res = ffmpeg_utils.concat_clips(clips, concat_out, fast_copy=False)
    assert res.exists()
    assert res.stat().st_size > 0

    # 2. Directly invoke FFmpeg filter_complex concat via subprocess to check exit code strictly
    direct_out = tmp_path / "direct_concat.mp4"
    inputs = []
    for c in clips:
        inputs.extend(["-i", str(c)])
    n = len(clips)
    filter_str = "".join(f"[{i}:v][{i}:a]" for i in range(n)) + f"concat=n={n}:v=1:a=1[outv][outa]"

    cmd = [
        ffmpeg, "-y",
        *inputs,
        "-filter_complex", filter_str,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "48000", "-ac", "2",
        str(direct_out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"filter_complex concat failed with exit code {proc.returncode} (stderr: {proc.stderr})"

    # 3. Probe concatenated results
    for out_file in (concat_out, direct_out):
        probe = ffmpeg_utils.probe_video(out_file)
        vstream = next(s for s in probe["streams"] if s["codec_type"] == "video")
        astream = next(s for s in probe["streams"] if s["codec_type"] == "audio")

        assert vstream["width"] == 1080
        assert vstream["height"] == 1920
        assert vstream.get("sample_aspect_ratio", "1:1") in ("1:1", None)
        assert vstream.get("display_aspect_ratio", "9:16") in ("9:16", None)
        assert vstream["r_frame_rate"] == "30/1"

        assert astream["sample_rate"] == "48000"
        assert astream["channels"] == 2

        dur = float(probe["format"]["duration"])
        expected_dur = 6 * 1.2  # 7.2s
        assert dur == pytest.approx(expected_dur, abs=0.2)


# ===========================================================================
# 3. Full VideoEditor Integration (Punch-In + J/L Split Edits)
# ===========================================================================

def test_empirical_editor_punch_in_with_split_edits(tmp_path: Path):
    """
    Empirical Integration Test:
    Run VideoEditor.execute_plan() with 4 clips combining:
    - Alternating 1.0x wide and 1.15x discrete punch-in zoom
    - J-cut audio lead (0.4s) on clip 2
    - L-cut audio tail (0.4s) on clip 3
    Verify resulting container has 1:1 SAR, 9:16 DAR, duration ~8.0s,
    and FFT spectral verification of mixed audio lead/tail.
    """
    inter_dir = tmp_path / "inter"
    inter_dir.mkdir(parents=True, exist_ok=True)

    src1 = _create_synthetic_test_video(tmp_path / "src1.mp4", duration=6.0, frequency=440)
    src2 = _create_synthetic_test_video(tmp_path / "src2.mp4", duration=6.0, frequency=1000)

    plan = EditingPlan(
        plan_id="p_punch_split",
        job_id="j_punch_split",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=8.0,
        hook_summary="hook",
        edit_rationale="punch-in and split-edit integration",
        clips=[
            # Clip 0: 1.0x wide, 0.0 - 2.0
            Clip(clip_id="c0", source_id="s1", source_start=0.0, source_end=2.0, order=0, reason="wide1"),
            # Clip 1: 1.15x punch-in zoom with J-cut audio lead 0.4s, 1.0 - 3.0
            Clip(
                clip_id="c1",
                source_id="s2",
                source_start=1.0,
                source_end=3.0,
                order=1,
                reason="punch1",
                zoom=ZoomEffect(enabled=True, factor=1.15),
                audio_lead=0.4,
            ),
            # Clip 2: 1.0x wide with L-cut audio tail 0.4s, 2.0 - 4.0
            Clip(
                clip_id="c2",
                source_id="s1",
                source_start=2.0,
                source_end=4.0,
                order=2,
                reason="wide2",
                audio_tail=0.4,
            ),
            # Clip 3: 1.15x punch-in zoom, 3.0 - 5.0
            Clip(
                clip_id="c3",
                source_id="s2",
                source_start=3.0,
                source_end=5.0,
                order=3,
                reason="punch2",
                zoom=ZoomEffect(enabled=True, factor=1.15),
            ),
        ],
        audio_settings=AudioSettings(normalize=False, fade_in=0.0, fade_out=0.0),
    )

    editor = VideoEditor()
    out_video = editor.execute_plan(
        plan=plan,
        source_path={"s1": src1, "s2": src2},
        job_dirs={"intermediate": inter_dir},
    )

    assert out_video.exists()
    assert out_video.stat().st_size > 0

    probe = ffmpeg_utils.probe_video(out_video)
    vstream = next(s for s in probe["streams"] if s["codec_type"] == "video")
    astream = next(s for s in probe["streams"] if s["codec_type"] == "audio")

    # Geometry & Aspect Ratio
    assert vstream["width"] == 1080
    assert vstream["height"] == 1920
    assert vstream.get("sample_aspect_ratio", "1:1") in ("1:1", None)
    assert vstream.get("display_aspect_ratio", "9:16") in ("9:16", None)
    assert vstream["r_frame_rate"] == "30/1"

    # Container Duration: 4 clips x 2.0s = 8.0s
    total_dur = float(probe["format"]["duration"])
    assert total_dur == pytest.approx(8.0, abs=0.2)

    # Audio Spectral Verification
    # J-cut on c1 (audio_lead=0.4): incoming clip 2 audio (1000Hz) should be active during 1.7s - 1.95s
    freqs_j, mag_j = _extract_pcm_spectrum(out_video, 1.7, 1.95)
    idx_1000 = np.argmin(np.abs(freqs_j - 1000.0))
    idx_440 = np.argmin(np.abs(freqs_j - 440.0))
    power_1000 = float(mag_j[idx_1000])
    power_440 = float(mag_j[idx_440])

    assert power_1000 > 5000.0, f"J-cut audio lead missing! Power@1000Hz={power_1000:.1f}, Power@440Hz={power_440:.1f}"

    # L-cut on c2 (audio_tail=0.4): c2 audio (440Hz) should extend into c3 video (during 6.05s - 6.3s)
    # Clip video boundary: c0 (0-2s), c1 (2-4s), c2 (4-6s), c3 (6-8s).
    # c2 audio tail (440Hz) extends past 6.0s up to 6.4s.
    freqs_l, mag_l = _extract_pcm_spectrum(out_video, 6.05, 6.3)
    idx_440_l = np.argmin(np.abs(freqs_l - 440.0))
    idx_1000_l = np.argmin(np.abs(freqs_l - 1000.0))
    power_440_l = float(mag_l[idx_440_l])
    power_1000_l = float(mag_l[idx_1000_l])

    assert power_440_l > 5000.0, f"L-cut audio tail missing! Power@440Hz={power_440_l:.1f}, Power@1000Hz={power_1000_l:.1f}"
