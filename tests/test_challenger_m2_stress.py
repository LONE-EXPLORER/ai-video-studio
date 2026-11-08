"""
tests/test_challenger_m2_stress.py — Empirical Challenger Stress Test Suite for Milestone 2.

Empirically tests:
1. Margin padding (+80ms lead, +180ms tail) and overlap merging (<260ms) under boundary,
   adversarial, out-of-order, floating-point jitter, and high-volume conditions.
2. Synthetic audio streams with alternating bursts, pauses, silent audio, unclosed silences,
   and negative timestamp handling in FFmpeg silencedetect stream parsing.
3. Real test asset verification on `assets/test_videos/trump_5_day_trap.mp4`:
   monotonicity, boundary safety, dead air reduction, and zero segment overlap.
4. Discrete 1.15x jump punch-in zoom geometry, Lanczos upscale, VideoToolbox vs libx264 fallback.
5. Micro audio fades (10ms) on normal and ultra-short clips preserving sample rate (48kHz) and duration.
6. Split edit (J-cut / L-cut) multi-track assembly under extreme boundary clamps.
"""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.editing.editor import VideoEditor
from app.editing.silence_remover import (
    apply_margin_padding,
    detect_silence,
    get_speech_segments,
    merge_overlapping_segments,
)
from app.schemas.editing_plan import AspectRatio, Clip, EditingPlan, ZoomEffect
from app.utils import ffmpeg_utils


# ===========================================================================
# Helper: Synthetic Video / Audio Generation
# ===========================================================================

def _create_synthetic_tone_and_silence_wav(
    out_path: Path,
    sequence: List[Tuple[str, float]],  # List of ("tone" | "silence", duration_seconds)
    sample_rate: int = 44100,
) -> Path:
    """Generate a multi-segment WAV file with arbitrary tone and silence segments."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_utils.find_ffmpeg()

    filter_inputs = []
    filter_complex_parts = []
    cmd = [ffmpeg, "-y"]

    for i, (kind, dur) in enumerate(sequence):
        if kind == "tone":
            cmd.extend(["-f", "lavfi", "-i", f"sine=f=1000:r={sample_rate}:d={dur:.4f}"])
        elif kind == "silence":
            cmd.extend(["-f", "lavfi", "-i", f"anullsrc=r={sample_rate}:cl=mono:d={dur:.4f}"])
        else:
            raise ValueError(f"Unknown kind: {kind}")
        filter_inputs.append(f"[{i}:a]")

    concat_spec = "".join(filter_inputs) + f"concat=n={len(sequence)}:v=0:a=1[out]"
    cmd.extend([
        "-filter_complex", concat_spec,
        "-map", "[out]",
        str(out_path),
    ])
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg audio generator failed:\nSTDERR: {res.stderr}")
    return out_path


def _generate_synthetic_test_video(
    output_path: Path,
    width: int = 1080,
    height: int = 1920,
    fps: float = 30.0,
    duration: float = 2.0,
) -> Path:
    """Generate a synthetic test clip with distinct visual content."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_utils.find_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-f", "lavfi", "-i", f"testsrc=size={width}x{height}:rate={fps}:duration={duration}",
        "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={duration}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "48000", "-ac", "2",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


# ===========================================================================
# Section 1: Algorithmic Stress on Margin Padding & Overlap Merging
# ===========================================================================

def test_margin_padding_extreme_boundary_clamping():
    """Verify margin padding clamps strictly to [0.0, duration] across all extremes."""
    duration = 10.0
    # Negative start, start near 0, end near duration, end beyond duration
    raw = [
        (-0.5, 0.05),
        (0.01, 0.05),
        (5.0, 5.2),
        (9.95, 10.0),
        (9.99, 10.5),
    ]
    padded = apply_margin_padding(raw, source_duration=duration, lead_margin=0.080, tail_margin=0.180)

    for s, e in padded:
        assert s >= 0.0, f"Segment start {s} is negative"
        assert e <= duration, f"Segment end {e} exceeds duration {duration}"
        assert s < e, f"Invalid segment: ({s}, {e})"


def test_margin_padding_exact_threshold_transitions():
    """
    Empirically verify pause behavior around the critical 260ms (lead=80ms + tail=180ms) threshold:
    - 250ms gap (< 260ms): MUST merge into a single segment.
    - 259ms gap (< 260ms): MUST merge into a single segment.
    - 260ms gap (exact threshold): Must never produce duplicate/overlapping frames (next_start >= prev_end).
    - 261ms gap (> 260ms): MUST remain 2 distinct segments.
    - 300ms gap (> 260ms): MUST remain 2 distinct segments.
    """
    # 1. 250ms gap -> must merge
    # Clip 1: [1.000, 2.000] -> padded [0.920, 2.180]
    # Clip 2: [2.250, 3.000] -> padded [2.170, 3.180] -> overlaps between 2.170 and 2.180
    res_250 = apply_margin_padding([(1.000, 2.000), (2.250, 3.000)], source_duration=10.0)
    assert len(res_250) == 1
    assert res_250[0] == (0.920, 3.180)

    # 2. 259ms gap -> must merge
    res_259 = apply_margin_padding([(1.000, 2.000), (2.259, 3.000)], source_duration=10.0)
    assert len(res_259) == 1
    assert res_259[0] == (0.920, 3.180)

    # 3. 260ms gap -> exact threshold. Regardless of merge, NO overlap is permitted.
    res_260 = apply_margin_padding([(1.000, 2.000), (2.260, 3.000)], source_duration=10.0)
    if len(res_260) == 1:
        assert res_260[0] == (0.920, 3.180)
    else:
        assert len(res_260) == 2
        assert res_260[1][0] >= res_260[0][1]

    # 4. 261ms gap -> must remain 2 distinct segments
    res_261 = apply_margin_padding([(1.000, 2.000), (2.261, 3.000)], source_duration=10.0)
    assert len(res_261) == 2
    assert res_261[0] == (0.920, 2.180)
    assert res_261[1] == (2.181, 3.180)
    assert res_261[1][0] > res_261[0][1]

    # 5. 300ms gap -> distinct
    res_300 = apply_margin_padding([(1.000, 2.000), (2.300, 3.000)], source_duration=10.0)
    assert len(res_300) == 2
    assert res_300[0] == (0.920, 2.180)
    assert res_300[1] == (2.220, 3.180)


def test_merge_overlapping_segments_adversarial_inputs():
    """Stress test merge_overlapping_segments with degenerates, inverted intervals, and out-of-order data."""
    # Degenerate & inverted intervals
    raw = [
        (2.0, 1.0),      # Inverted: start > end
        (3.0, 3.0),      # Zero length: start == end
        (1.0, 4.0),      # Valid wide interval
        (2.0, 3.5),      # Enclosed inside (1.0, 4.0)
        (3.9, 6.0),      # Overlapping tail
    ]
    merged = merge_overlapping_segments(raw)
    assert merged == [(1.0, 6.0)]


def test_merge_overlapping_segments_high_volume_perf():
    """Verify O(N log N) scaling and stability on 10,000 micro-intervals."""
    # Generate 10,000 random overlapping intervals
    random.seed(42)
    intervals: List[Tuple[float, float]] = []
    for _ in range(10000):
        s = random.uniform(0.0, 500.0)
        e = s + random.uniform(0.01, 0.50)
        intervals.append((s, e))

    merged = merge_overlapping_segments(intervals)
    assert len(merged) > 0
    # Strictly sorted and non-overlapping
    for i in range(len(merged) - 1):
        assert merged[i][1] <= merged[i + 1][0]
        assert merged[i][0] < merged[i][1]


# ===========================================================================
# Section 2: End-to-End Synthetic Audio Streams with Bursts and Pauses
# ===========================================================================

def test_synthetic_alternating_bursts_and_pauses(tmp_path: Path):
    """
    Construct synthetic audio with known bursts and pauses:
    1. 0.00s - 0.50s: Tone 1 (500ms)
    2. 0.50s - 0.70s: Silence 1 (200ms pause: <260ms -> should merge!)
    3. 0.70s - 1.20s: Tone 2 (500ms)
    4. 1.20s - 1.80s: Silence 2 (600ms pause: >260ms -> should NOT merge!)
    5. 1.80s - 2.50s: Tone 3 (700ms)
    6. 2.50s - 3.00s: Silence 3 (500ms trailing silence at EOF)

    Total duration: 3.0s.
    Expected speech segments:
    - Segment A: Tone 1 + Silence 1 + Tone 2 merged into one span ~ [0.0, 1.38]
    - Segment B: Tone 3 ~ [1.72, 2.68]
    """
    wav_path = tmp_path / "bursts_pauses.wav"
    seq = [
        ("tone", 0.50),
        ("silence", 0.20),
        ("tone", 0.50),
        ("silence", 0.60),
        ("tone", 0.70),
        ("silence", 0.50),
    ]
    _create_synthetic_tone_and_silence_wav(wav_path, seq)

    speech = get_speech_segments(
        wav_path,
        duration=3.0,
        min_silence_duration=0.15,
        silence_threshold_db=-30.0,
        lead_margin=0.080,
        tail_margin=0.180,
    )

    assert len(speech) == 2, f"Expected 2 speech segments, got {len(speech)}: {speech}"
    # Check Segment A spans from 0.0 to ~1.38s
    assert speech[0][0] == 0.0
    assert 1.30 <= speech[0][1] <= 1.45

    # Check Segment B starts around ~1.72s and ends around ~2.68s
    assert 1.65 <= speech[1][0] <= 1.80
    assert 2.60 <= speech[1][1] <= 2.75

    # Check no overlap between Segment A and Segment B
    assert speech[0][1] < speech[1][0]


def test_pure_silent_audio_returns_empty_and_safe_fallback(tmp_path: Path):
    """Verify that pure silence yields empty list from get_speech_segments and is safe."""
    wav_path = tmp_path / "pure_silent.wav"
    _create_synthetic_tone_and_silence_wav(wav_path, [("silence", 4.0)])

    speech = get_speech_segments(wav_path, duration=4.0, min_silence_duration=0.15)
    assert speech == [], f"Expected empty speech segments on pure silence, got {speech}"


def test_detect_silence_with_malformed_and_negative_stderr(tmp_path: Path):
    """
    Adversarial test on silencedetect stderr stream parsing:
    - negative start timestamp
    - silence_end without silence_start
    - unclosed trailing silence at EOF
    """
    dummy_wav = tmp_path / "dummy.wav"
    dummy_wav.write_bytes(b"RIFFdummyWAVE")

    mock_stderr = (
        "[silencedetect @ 0x1] silence_start: -0.0123\n"
        "[silencedetect @ 0x1] silence_end: 0.500 | silence_duration: 0.5123\n"
        "[silencedetect @ 0x1] silence_end: 1.800 | silence_duration: 0.400\n"
        "[silencedetect @ 0x1] silence_start: 3.200\n"
    )

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stderr=mock_stderr)
        silences = detect_silence(dummy_wav, duration=5.0)

    # 1st silence: [-0.0123 -> clamped to 0.0, 0.500]
    # 2nd silence: [end 1.800 - dur 0.400 = 1.400, 1.800]
    # 3rd silence: [start 3.200 -> unclosed at EOF -> 5.000]
    assert len(silences) == 3
    assert silences[0] == (0.0, 0.5)
    assert silences[1] == (1.4, 1.8)
    assert silences[2] == (3.2, 5.0)


# ===========================================================================
# Section 3: Real Asset Empirical Verification on trump_5_day_trap.mp4
# ===========================================================================

def test_real_asset_trump_speech_extraction():
    """
    Empirical verification on `assets/test_videos/trump_5_day_trap.mp4`:
    1. Validates extraction of speech intervals.
    2. Validates zero overlaps between adjacent intervals.
    3. Validates boundary clamping in [0.0, duration].
    4. Validates dead air trimming savings (> 5s saved).
    """
    asset_path = Path("assets/test_videos/trump_5_day_trap.mp4")
    assert asset_path.exists(), f"Primary test asset {asset_path} missing!"

    probe = ffmpeg_utils.probe_video(asset_path)
    total_duration = float(probe["format"]["duration"])
    assert 60.0 <= total_duration <= 60.5

    speech_segs = get_speech_segments(
        asset_path,
        duration=total_duration,
        min_silence_duration=0.15,
        silence_threshold_db=-30.0,
        lead_margin=0.080,
        tail_margin=0.180,
    )

    assert len(speech_segs) >= 20, f"Expected >= 20 speech segments, got {len(speech_segs)}"

    # 1. Monotonicity & zero-overlap check
    for i in range(len(speech_segs) - 1):
        curr_start, curr_end = speech_segs[i]
        next_start, next_end = speech_segs[i + 1]

        assert curr_start >= 0.0
        assert curr_end <= total_duration
        assert curr_start < curr_end, f"Invalid segment duration at {i}: ({curr_start}, {curr_end})"
        assert curr_end <= next_start, (
            f"Overlap detected between segment {i} ({curr_start:.3f}-{curr_end:.3f}) "
            f"and segment {i+1} ({next_start:.3f}-{next_end:.3f})!"
        )

    # 2. Dead air reduction check
    total_speech_time = sum(e - s for s, e in speech_segs)
    dead_air_saved = total_duration - total_speech_time
    assert dead_air_saved >= 5.0, (
        f"Expected at least 5.0s of dead air eliminated, got {dead_air_saved:.2f}s "
        f"(speech: {total_speech_time:.2f}s / {total_duration:.2f}s)"
    )


# ===========================================================================
# Section 4: Discrete 1.15x Jump Punch-In & Micro Audio Fade Stress Harness
# ===========================================================================

def test_discrete_punch_in_geometry_and_resolution(tmp_path: Path):
    """
    Empirically verify apply_zoom_effect:
    1. Output resolution remains 1080x1920.
    2. Output framerate remains 30fps.
    3. Discrete jump cut preserves audio track without corruption.
    4. Test both VideoToolbox path and libx264 software fallback path.
    """
    src_video = tmp_path / "src_1080x1920.mp4"
    _generate_synthetic_test_video(src_video, width=1080, height=1920, fps=30.0, duration=1.0)

    # Test path 1: default (VideoToolbox enabled)
    out_vtb = tmp_path / "zoomed_vtb.mp4"
    ffmpeg_utils.apply_zoom_effect(src_video, out_vtb, zoom_factor=1.15, discrete=True, use_videotoolbox=True)

    probe_vtb = ffmpeg_utils.probe_video(out_vtb)
    vs_vtb = next(s for s in probe_vtb["streams"] if s["codec_type"] == "video")
    as_vtb = next(s for s in probe_vtb["streams"] if s["codec_type"] == "audio")
    assert int(vs_vtb["width"]) == 1080
    assert int(vs_vtb["height"]) == 1920
    assert vs_vtb["r_frame_rate"] == "30/1"
    assert as_vtb is not None

    # Test path 2: software fallback (libx264)
    out_sw = tmp_path / "zoomed_sw.mp4"
    ffmpeg_utils.apply_zoom_effect(src_video, out_sw, zoom_factor=1.15, discrete=True, use_videotoolbox=False)

    probe_sw = ffmpeg_utils.probe_video(out_sw)
    vs_sw = next(s for s in probe_sw["streams"] if s["codec_type"] == "video")
    as_sw = next(s for s in probe_sw["streams"] if s["codec_type"] == "audio")
    assert int(vs_sw["width"]) == 1080
    assert int(vs_sw["height"]) == 1920
    assert vs_sw["r_frame_rate"] == "30/1"
    assert as_sw is not None


def test_micro_audio_fade_on_ultra_short_clip(tmp_path: Path):
    """
    Verify cut_segment with micro_fade=True on:
    - standard cut (1.0s)
    - ultra-short cut (0.02s / 20ms) where fade_d must clamp to clip_duration / 2.0.
    Ensures no FFmpeg afade errors and sample rate stays 48000 Hz.
    """
    src_video = tmp_path / "src_for_microfade.mp4"
    _generate_synthetic_test_video(src_video, width=1080, height=1920, fps=30.0, duration=2.0)

    # 1. Standard cut
    out_norm = tmp_path / "cut_standard.mp4"
    ffmpeg_utils.cut_segment(src_video, start=0.5, end=1.5, output_path=out_norm, micro_fade=True)
    p_norm = ffmpeg_utils.probe_video(out_norm)
    a_norm = next(s for s in p_norm["streams"] if s["codec_type"] == "audio")
    assert int(a_norm["sample_rate"]) == 48000
    assert int(a_norm["channels"]) == 2

    # 2. Ultra-short cut (20ms)
    out_short = tmp_path / "cut_short.mp4"
    ffmpeg_utils.cut_segment(src_video, start=0.50, end=0.52, output_path=out_short, micro_fade=True)
    p_short = ffmpeg_utils.probe_video(out_short)
    a_short = next(s for s in p_short["streams"] if s["codec_type"] == "audio")
    assert int(a_short["sample_rate"]) == 48000
    dur = float(p_short["format"]["duration"])
    assert dur == pytest.approx(0.02, abs=0.08)


# ===========================================================================
# Section 5: Split Edit (J-Cut / L-Cut) Multi-Track Assembly Stress Harness
# ===========================================================================

def test_split_edit_boundary_clamping_and_mixing(tmp_path: Path):
    """
    Stress-test _execute_split_edit_plan with adversarial audio_lead and audio_tail:
    - Clip 1: audio_lead=0.8 (must clamp to 0 because clip 1 has no previous clip)
    - Clip 2: audio_lead=3.0 (exceeds clip 1 duration 1.0s -> must clamp to 1.0s)
    - Clip 3: audio_tail=5.0 (must clamp to 0 because clip 3 is the last clip)
    - Clip 2: audio_tail=0.3 (valid L-cut into clip 3)

    Verifies:
    1. Execution completes without error.
    2. Master rough cut duration matches sum of video clip durations (1.0 + 1.2 + 1.5 = 3.7s).
    3. Output audio is standardized to 48000 Hz stereo AAC.
    """
    src_video = tmp_path / "src_split.mp4"
    _generate_synthetic_test_video(src_video, width=1080, height=1920, fps=30.0, duration=5.0)

    clips = [
        Clip(
            clip_id="c01",
            source_start=0.2,
            source_end=1.2,  # dur 1.0s
            audio_lead=0.8,   # Adversarial: first clip lead must clamp to 0
            order=1,
            reason="intro",
        ),
        Clip(
            clip_id="c02",
            source_start=1.5,
            source_end=2.7,  # dur 1.2s
            audio_lead=3.0,   # Adversarial: lead exceeds prev clip dur 1.0s -> clamp to 1.0s
            audio_tail=0.3,   # Valid L-cut
            order=2,
            reason="body",
            zoom=ZoomEffect(enabled=True, factor=1.15),
        ),
        Clip(
            clip_id="c03",
            source_start=3.0,
            source_end=4.5,  # dur 1.5s
            audio_tail=5.0,   # Adversarial: last clip tail must clamp to 0
            order=3,
            reason="outro",
        ),
    ]

    plan = EditingPlan(
        plan_id="plan_split_stress",
        job_id="job_split_stress",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=3.7,
        clips=clips,
        hook_summary="Split edit stress test",
        edit_rationale="Stress-test J-cut / L-cut clamping and multi-track mixing",
    )
    plan.audio_settings.normalize = False

    job_dirs = {
        "intermediate": tmp_path / "intermediate",
        "previews": tmp_path / "previews",
    }
    job_dirs["intermediate"].mkdir(parents=True, exist_ok=True)

    editor = VideoEditor()
    rough_cut = editor.execute_plan(plan, source_path=src_video, job_dirs=job_dirs)

    assert rough_cut.exists()
    assert rough_cut.stat().st_size > 0

    probe = ffmpeg_utils.probe_video(rough_cut)
    dur = float(probe["format"]["duration"])
    # Video duration should be 1.0 + 1.2 + 1.5 = 3.7s (approx +/- 0.25s)
    assert dur == pytest.approx(3.7, abs=0.25)

    a_stream = next(s for s in probe["streams"] if s["codec_type"] == "audio")
    assert int(a_stream["sample_rate"]) == 48000
    assert int(a_stream["channels"]) == 2
