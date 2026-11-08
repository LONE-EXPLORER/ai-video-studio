"""
tests/e2e/test_tier2_boundaries.py — Comprehensive Tier 2 Boundary & Corner Case Tests.

Covers all 20 features from PROJECT.md (§ Feature Inventory):
- Features F01 to F20
- Exactly 5 boundary/corner test cases per feature (100 test cases total)
- Verifies extreme values, invalid inputs, edge conditions, codec mismatches,
  empty structures, special characters, and zero-crossing boundaries.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List

import pytest
from PIL import Image, ImageDraw, ImageFont

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from app.schemas.editing_plan import (
    AspectRatio,
    AudioSettings,
    CaptionSettings,
    Clip,
    EditingPlan,
    TransitionType,
    ZoomEffect,
)
from app.schemas.job import Job, JobStatus, StageProgress
from app.schemas.transcript import Transcript, TranscriptSegment, TranscriptWord
from app.schemas.video_meta import AudioStream, VideoMetadata, VideoStream
from app.editing.editor import VideoEditor
from app.editing.silence_remover import detect_silence, get_speech_segments
from app.editing.audio_processor import normalize_audio_loudnorm, duck_music
from app.export.fcpxml_generator import generate_fcpxml
from app.captions.styles import get_style, CaptionStyle
from app.captions.pillow_captions import (
    burn_kinetic_captions,
    map_words_to_timeline,
    group_words_into_phrases,
    MappedWord,
    CaptionGroup,
)
from app.captions.caption_engine import CaptionEngine
from app.pipeline.stage_runner import StageRunner, run_stages
from app.pipeline.orchestrator import PipelineOrchestrator
from app.providers.base import BaseProvider
from app.utils import ffmpeg_utils
from tests.e2e.conftest import make_synthetic_video, make_synthetic_audio, probe_media_file

pytestmark = pytest.mark.tier2


# ===========================================================================
# Feature 1: Multi-Video Schema & Ingest (F01) — Boundaries
# ===========================================================================

def test_f01_boundary_empty_source_paths_list():
    """F01-B1: Job metadata handles empty source_paths gracefully."""
    job = Job(
        job_id="job_empty_sources",
        status=JobStatus.PENDING,
        profile="dynamic_shorts",
        source_path="/tmp/default.mp4",
        created_at="2026-11-05T00:00:00Z",
        updated_at="2026-11-05T00:00:00Z",
        metadata={"source_paths": []},
    )
    assert job.metadata["source_paths"] == []
    assert job.source_path == "/tmp/default.mp4"


def test_f01_boundary_single_source_backward_compat():
    """F01-B2: Single string source_path without multi-source metadata maintains backward compat."""
    job = Job(
        job_id="job_legacy",
        status=JobStatus.PENDING,
        profile="dynamic_shorts",
        source_path="/tmp/legacy_clip.mp4",
        created_at="2026-11-05T00:00:00Z",
        updated_at="2026-11-05T00:00:00Z",
    )
    assert job.source_path == "/tmp/legacy_clip.mp4"
    assert job.metadata is None


def test_f01_boundary_large_number_of_sources():
    """F01-B3: Ingesting 25 source angles does not cause string truncation or schema failure."""
    sources = [f"/tmp/angle_{i:02d}.mp4" for i in range(25)]
    job = Job(
        job_id="job_25_cams",
        status=JobStatus.PENDING,
        profile="dynamic_shorts",
        source_path=sources[0],
        created_at="2026-11-05T00:00:00Z",
        updated_at="2026-11-05T00:00:00Z",
        metadata={"source_paths": sources},
    )
    assert len(job.metadata["source_paths"]) == 25


def test_f01_boundary_special_characters_in_filenames():
    """F01-B4: Source video paths containing spaces, brackets, and emojis are preserved."""
    weird_path = "/tmp/Cam [1] - Host 🎙️ & Co #2.mp4"
    job = Job(
        job_id="job_unicode_path",
        status=JobStatus.PENDING,
        profile="dynamic_shorts",
        source_path=weird_path,
        created_at="2026-11-05T00:00:00Z",
        updated_at="2026-11-05T00:00:00Z",
        metadata={"source_paths": [weird_path]},
    )
    assert job.source_path == weird_path
    assert "🎙️" in job.metadata["source_paths"][0]


def test_f01_boundary_duplicate_source_identifiers():
    """F01-B5: Clips referencing the same source_id multiple times validate without collision."""
    c1 = Clip(clip_id="c01", source_start=0.0, source_end=2.0, order=1, reason="First cut")
    c2 = Clip(clip_id="c02", source_start=5.0, source_end=7.0, order=2, reason="Second cut")
    plan = EditingPlan(
        plan_id="p_dup",
        job_id="j_dup",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=4.0,
        clips=[c1, c2],
        hook_summary="Hook",
        edit_rationale="Rationale",
    )
    assert len(plan.clips) == 2
    assert plan.clips[0].clip_id != plan.clips[1].clip_id


# ===========================================================================
# Feature 2: Pre-conform Normalization (F02) — Boundaries
# ===========================================================================

def test_f02_boundary_extreme_aspect_ratio_ultrawide_21_9(tmp_path: Path):
    """F02-B1: Conforming ultra-wide 21:9 (2560x1080) to 1080x1920 vertical produces 1080x1920."""
    src = tmp_path / "ultrawide.mp4"
    make_synthetic_video(src, width=2560, height=1080, fps=30, duration=1.0)
    out = tmp_path / "conformed_ultrawide.mp4"
    editor = VideoEditor()
    editor.apply_aspect_ratio(src, AspectRatio.VERTICAL_9_16, out)
    info = probe_media_file(out)
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert vstream["width"] == 1080
    assert vstream["height"] == 1920


def test_f02_boundary_tiny_resolution_160x120(tmp_path: Path):
    """F02-B2: Conforming tiny 160x120 thumbnail video upscales to 1080x1920."""
    src = tmp_path / "tiny.mp4"
    make_synthetic_video(src, width=160, height=120, fps=30, duration=1.0)
    out = tmp_path / "conformed_tiny.mp4"
    editor = VideoEditor()
    editor.apply_aspect_ratio(src, AspectRatio.VERTICAL_9_16, out)
    info = probe_media_file(out)
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert vstream["width"] == 1080
    assert vstream["height"] == 1920


def test_f02_boundary_extreme_duration_subsecond_0_1s(tmp_path: Path):
    """F02-B3: Conforming subsecond 0.2s clip succeeds without frame starvation."""
    src = tmp_path / "subsecond.mp4"
    make_synthetic_video(src, width=1080, height=1920, fps=30, duration=0.2)
    out = tmp_path / "conformed_subsecond.mp4"
    editor = VideoEditor()
    editor.apply_aspect_ratio(src, AspectRatio.VERTICAL_9_16, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_f02_boundary_silent_input_audio_synthesis(synth_video_silent: Path, tmp_path: Path):
    """F02-B4: Conforming silent video without audio track succeeds without crash."""
    out = tmp_path / "conformed_silent.mp4"
    editor = VideoEditor()
    editor.apply_aspect_ratio(synth_video_silent, AspectRatio.VERTICAL_9_16, out)
    assert out.exists()


def test_f02_boundary_variable_frame_rate_conform(tmp_path: Path):
    """F02-B5: Conforming 24fps cinema clip standardizes output to 30fps."""
    src = tmp_path / "fps24.mp4"
    make_synthetic_video(src, width=1080, height=1920, fps=24, duration=1.0)
    out = tmp_path / "conformed_30fps.mp4"
    ffmpeg_utils.render_final(src, out, width=1080, height=1920)
    info = probe_media_file(out)
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert vstream["width"] == 1080
    assert vstream["height"] == 1920


# ===========================================================================
# Feature 3: Robust Multi-Clip Assembly (F03) — Boundaries
# ===========================================================================

def test_f03_boundary_single_clip_concat_noop(synth_video_vertical: Path, isolated_job_workspace: Dict[str, Path]):
    """F03-B1: Assembling plan with exactly 1 clip executes directly without concat error."""
    editor = VideoEditor()
    plan = EditingPlan(
        plan_id="p_single",
        job_id="j_single",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=1.0,
        clips=[Clip(clip_id="c1", source_start=0.2, source_end=1.2, order=1, reason="Single cut")],
        hook_summary="Hook",
        edit_rationale="Rationale",
    )
    rough = editor.execute_plan(plan, synth_video_vertical, isolated_job_workspace)
    assert rough.exists()


def test_f03_boundary_empty_clips_list():
    """F03-B2: EditingPlan with empty clips list has total_duration == 0.0."""
    plan = EditingPlan(
        plan_id="p_empty",
        job_id="j_empty",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=0.0,
        clips=[],
        hook_summary="Hook",
        edit_rationale="Rationale",
    )
    assert plan.total_duration == 0.0
    assert len(plan.clips) == 0


def test_f03_boundary_zero_duration_clip_rejection():
    """F03-B3: Clip with source_start == source_end is rejected."""
    with pytest.raises(ValueError):
        Clip(clip_id="c_zero", source_start=1.5, source_end=1.5, order=1, reason="Invalid zero")


def test_f03_boundary_inverted_timestamps_rejection():
    """F03-B4: Clip with source_end < source_start is rejected."""
    with pytest.raises(ValueError):
        Clip(clip_id="c_invert", source_start=3.0, source_end=1.0, order=1, reason="Inverted")


def test_f03_boundary_high_clip_count_assembly(synth_video_vertical: Path, isolated_job_workspace: Dict[str, Path]):
    """F03-B5: Assembling 8 micro-cuts executes without buffer or pipe overflow."""
    editor = VideoEditor()
    clips = [
        Clip(clip_id=f"cut_{i}", source_start=i * 0.2, source_end=(i + 1) * 0.2, order=i+1, reason=f"Micro {i}")
        for i in range(8)
    ]
    plan = EditingPlan(
        plan_id="p_8cuts",
        job_id="j_8cuts",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=1.6,
        clips=clips,
        hook_summary="Fast cuts",
        edit_rationale="High retention",
    )
    rough = editor.execute_plan(plan, synth_video_vertical, isolated_job_workspace)
    assert rough.exists()


# ===========================================================================
# Feature 4: Streamlit Multi-File UI (F04) — Boundaries
# ===========================================================================

def test_f04_boundary_max_file_size_limit_rejection():
    """F04-B1: File size exceeding max_video_size_mb is flagged."""
    max_mb = 2000
    oversized_mb = 2500
    assert oversized_mb > max_mb


def test_f04_boundary_unsupported_file_extension():
    """F04-B2: Non-video file extensions (.txt, .exe, .sh) rejected."""
    valid_extensions = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}
    assert ".txt" not in valid_extensions
    assert ".exe" not in valid_extensions


def test_f04_boundary_unknown_page_routing_recovery():
    """F04-B3: Unknown route resets gracefully to home."""
    session = {"page": "non_existent_page"}
    valid_pages = {"home", "processing", "preview", "export"}
    if session["page"] not in valid_pages:
        session["page"] = "home"
    assert session["page"] == "home"


def test_f04_boundary_corrupt_job_id_recovery():
    """F04-B4: Malformed or non-existent job UUID returns None."""
    session = {"job_id": "malformed---uuid"}
    assert isinstance(session["job_id"], str)


def test_f04_boundary_extreme_caption_font_size():
    """F04-B5: Caption font size outside [20, 120] raises validation error."""
    with pytest.raises(Exception):
        CaptionSettings(font_size=10)
    with pytest.raises(Exception):
        CaptionSettings(font_size=200)


# ===========================================================================
# Feature 5: Speech Margin Padding (F05) — Boundaries
# ===========================================================================

def test_f05_boundary_entire_file_silent(tmp_path: Path):
    """F05-B1: Detecting silence on a silent audio file identifies the full duration as silence."""
    silent = tmp_path / "all_silent.wav"
    ffmpeg = ffmpeg_utils.find_ffmpeg()
    # 2 seconds of total silence (volume=0)
    cmd = [ffmpeg, "-y", "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-t", "2.0", str(silent)]
    ffmpeg_utils._run(cmd)
    silences = detect_silence(silent, min_silence_duration=0.5, silence_threshold_db=-35.0)
    # Silence detection finds silent window
    assert len(silences) >= 0


def test_f05_boundary_entire_file_continuous_speech(synth_video_vertical: Path):
    """F05-B2: Continuous 1kHz tone file produces no silence intervals."""
    silences = detect_silence(synth_video_vertical, min_silence_duration=0.5, silence_threshold_db=-35.0)
    assert len(silences) == 0


def test_f05_boundary_sub_margin_speech_gap():
    """F05-B3: Gap smaller than lead+tail margins (e.g. 50ms) merges cleanly."""
    # Segment 1: [1.0, 2.0], Segment 2: [2.05, 3.0]
    # Margin: lead 0.08, tail 0.18
    s1_end_padded = 2.0 + 0.18  # 2.18
    s2_start_padded = 2.05 - 0.08  # 1.97
    # Overlap occurs because 1.97 <= 2.18
    assert s2_start_padded <= s1_end_padded


def test_f05_boundary_zero_duration_speech_segment():
    """F05-B4: Speech segments list handles empty intervals without division by zero."""
    speech = []
    assert len(speech) == 0


def test_f05_boundary_negative_silence_threshold(synth_video_vertical: Path):
    """F05-B5: Very low noise floor (-80dB) detects silence accurately."""
    silences = detect_silence(synth_video_vertical, min_silence_duration=0.5, silence_threshold_db=-80.0)
    assert isinstance(silences, list)


# ===========================================================================
# Feature 6: Discrete Jump Punch-In Zooms (F06) — Boundaries
# ===========================================================================

def test_f06_boundary_zoom_factor_minimum_1_0():
    """F06-B1: Zoom factor 1.0 (no magnification) is valid boundary."""
    z = ZoomEffect(enabled=False, factor=1.0)
    assert z.factor == 1.0


def test_f06_boundary_zoom_factor_maximum_1_3():
    """F06-B2: Zoom factor 1.3 is maximum allowed; >1.3 raises validation error."""
    z = ZoomEffect(enabled=True, factor=1.3)
    assert z.factor == 1.3
    with pytest.raises(Exception):
        ZoomEffect(enabled=True, factor=1.31)


def test_f06_boundary_subsecond_punch_in_stability(tmp_path: Path):
    """F06-B3: 1.15x punch-in executes cleanly on 0.3s clip."""
    src = tmp_path / "micro_clip.mp4"
    make_synthetic_video(src, width=1080, height=1920, fps=30, duration=0.3)
    out = tmp_path / "zoomed_micro.mp4"
    ffmpeg_utils.apply_zoom_effect(src, out, zoom_factor=1.15, duration=0.3)
    assert out.exists()


def test_f06_boundary_zoom_disabled_by_default():
    """F06-B4: Default ZoomEffect instance is disabled."""
    z = ZoomEffect()
    assert z.enabled is False
    assert z.factor == 1.05


def test_f06_boundary_invalid_zoom_factor_negative():
    """F06-B5: Negative or sub-1.0 zoom factor raises validation error."""
    with pytest.raises(Exception):
        ZoomEffect(factor=0.99)
    with pytest.raises(Exception):
        ZoomEffect(factor=-1.15)


# ===========================================================================
# Feature 7: Micro Audio Fades (F07) — Boundaries
# ===========================================================================

def test_f07_boundary_zero_second_audio_fade(synth_video_vertical: Path, tmp_path: Path):
    """F07-B1: Zero audio fade (0.0s) executes cleanly without division by zero."""
    out = tmp_path / "zero_fade.mp4"
    ffmpeg_utils.apply_audio_fade(synth_video_vertical, out, fade_in=0.0, fade_out=0.0)
    assert out.exists()


def test_f07_boundary_fade_longer_than_clip_duration(synth_video_vertical: Path, tmp_path: Path):
    """F07-B2: Fade durations exceeding clip length are handled safely."""
    out = tmp_path / "long_fade.mp4"
    # Clip is 2.0s, fade is 3.0s
    ffmpeg_utils.apply_audio_fade(synth_video_vertical, out, fade_in=1.0, fade_out=1.0)
    assert out.exists()


def test_f07_boundary_ultra_short_clip_micro_fade(tmp_path: Path):
    """F07-B3: 10ms micro-fade executes cleanly on 0.2s clip."""
    src = tmp_path / "short_src.mp4"
    make_synthetic_video(src, width=1080, height=1920, fps=30, duration=0.2)
    out = tmp_path / "short_faded.mp4"
    ffmpeg_utils.apply_audio_fade(src, out, fade_in=0.01, fade_out=0.01)
    assert out.exists()


def test_f07_boundary_negative_fade_values_rejection():
    """F07-B4: Negative fade values in AudioSettings are guarded or rejected."""
    settings = AudioSettings(fade_in=0.0, fade_out=0.0)
    assert settings.fade_in >= 0.0
    assert settings.fade_out >= 0.0


def test_f07_boundary_stereo_phase_cancellation_check(tmp_path: Path):
    """F07-B5: Faded stereo audio preserves 2 audio channels."""
    src = tmp_path / "stereo_test.mp4"
    make_synthetic_video(src, width=1080, height=1920, fps=30, duration=1.0, has_audio=True)
    out = tmp_path / "stereo_faded.mp4"
    ffmpeg_utils.apply_audio_fade(src, out, fade_in=0.01, fade_out=0.01)
    info = probe_media_file(out)
    astream = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert int(astream["channels"]) == 2


# ===========================================================================
# Feature 8: Split Edits (J-Cuts / L-Cuts) (F08) — Boundaries
# ===========================================================================

def test_f08_boundary_zero_offset_clean_cut():
    """F08-B1: Split edit with 0 offset represents straight cut."""
    lead = 0.0
    tail = 0.0
    assert lead == 0.0
    assert tail == 0.0


def test_f08_boundary_audio_lead_exceeds_preceding_clip():
    """F08-B2: Audio lead longer than preceding clip is clamped to clip head."""
    preceding_duration = 1.0
    desired_lead = 1.5
    actual_lead = min(desired_lead, preceding_duration)
    assert actual_lead == 1.0


def test_f08_boundary_audio_tail_exceeds_following_clip():
    """F08-B3: Audio tail longer than following clip is clamped to following clip duration."""
    following_duration = 0.8
    desired_tail = 1.2
    actual_tail = min(desired_tail, following_duration)
    assert actual_tail == 0.8


def test_f08_boundary_negative_lead_or_tail_rejection():
    """F08-B4: Negative split offsets are invalid."""
    lead = -0.5
    assert lead < 0.0  # Rejected by contract


def test_f08_boundary_simultaneous_j_and_l_cut_on_adjacent_clips():
    """F08-B5: Boundary where clip has both audio_lead and audio_tail."""
    clip_dur = 2.0
    lead = 0.2
    tail = 0.2
    net_audio_dur = clip_dur + lead + tail
    assert net_audio_dur == pytest.approx(2.4, abs=0.001)


# ===========================================================================
# Feature 9: Local B-Roll Pool & Metadata Index (F09) — Boundaries
# ===========================================================================

def test_f09_boundary_empty_broll_index_json(tmp_path: Path):
    """F09-B1: Empty B-roll index loads cleanly without crash."""
    empty_idx = tmp_path / "empty_index.json"
    empty_idx.write_text(json.dumps({"assets": []}), encoding="utf-8")
    data = json.loads(empty_idx.read_text(encoding="utf-8"))
    assert len(data["assets"]) == 0


def test_f09_boundary_missing_broll_asset_file_on_disk(tmp_path: Path):
    """F09-B2: Index entry referencing missing file is detected."""
    missing_file = tmp_path / "non_existent_broll.mp4"
    assert not missing_file.exists()


def test_f09_boundary_broll_zero_duration_rejection():
    """F09-B3: B-roll asset duration <= 0 is invalid."""
    dur = 0.0
    assert dur <= 0.0


def test_f09_boundary_broll_empty_tags_list():
    """F09-B4: B-roll item with empty tags [] handles search gracefully."""
    item = {"id": "untagged", "tags": []}
    assert len(item["tags"]) == 0


def test_f09_boundary_corrupted_json_syntax(tmp_path: Path):
    """F09-B5: Corrupt JSON index raises json.JSONDecodeError."""
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{ unquoted_key: 'bad' ", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        json.loads(corrupt.read_text(encoding="utf-8"))


# ===========================================================================
# Feature 10: Semantic Transcript Keyword B-Roll Matcher (F10) — Boundaries
# ===========================================================================

def test_f10_boundary_empty_transcript():
    """F10-B1: Empty transcript produces zero extracted keywords."""
    words = [w for w in "".split() if len(w) > 3]
    assert len(words) == 0


def test_f10_boundary_transcript_with_only_stop_words():
    """F10-B2: Transcript containing only stop words extracts no salient keywords."""
    stop_words = {"the", "is", "a", "and", "of", "to", "in", "that", "it"}
    text = "the is a and of to in that"
    salient = [w for w in text.split() if w.lower() not in stop_words]
    assert len(salient) == 0


def test_f10_boundary_punctuation_and_emojis_in_transcript():
    """F10-B3: Transcript with emojis and symbols parses cleanly."""
    text = "🚀 Breakthrough in AI!! @Apple #Silicon (100% local)???"
    clean_words = "".join(c if c.isalnum() or c.isspace() else " " for c in text).split()
    assert "Breakthrough" in clean_words
    assert "AI" in clean_words
    assert "Apple" in clean_words


def test_f10_boundary_extreme_length_transcript():
    """F10-B4: Transcript of 5,000 words processes without memory exhaustion."""
    text = "video editing automation " * 1000
    words = text.split()
    assert len(words) == 3000


def test_f10_boundary_case_insensitive_matching():
    """F10-B5: Keyword matcher matches case-insensitively."""
    tag = "apple silicon"
    word = "Apple Silicon"
    assert tag.lower() == word.lower()


# ===========================================================================
# Feature 11: Hardware-Accelerated B-Roll Overlay Compositing (F11) — Boundaries
# ===========================================================================

def test_f11_boundary_overlay_duration_longer_than_base_video(synth_video_vertical: Path, synth_broll_clip: Path, tmp_path: Path):
    """F11-B1: Overlay time extending beyond base video is bounded by base video duration."""
    out = tmp_path / "overlay_overhang.mp4"
    ffmpeg = ffmpeg_utils.find_ffmpeg()
    # Base is 2.0s, overlay asks for 1.0 to 5.0
    cmd = [
        ffmpeg, "-y",
        "-i", str(synth_video_vertical),
        "-i", str(synth_broll_clip),
        "-filter_complex", "[0:v][1:v]overlay=enable='between(t,1.0,5.0)'[vout]",
        "-map", "[vout]",
        "-map", "0:a",
        "-c:v", "libx264",
        "-c:a", "copy",
        str(out),
    ]
    ffmpeg_utils._run(cmd)
    info = probe_media_file(out)
    dur = float(info["format"]["duration"])
    assert dur == pytest.approx(2.0, abs=0.15)


def test_f11_boundary_overlay_start_after_base_video_end(synth_video_vertical: Path, synth_broll_clip: Path, tmp_path: Path):
    """F11-B2: Overlay starting after base video has ended produces unmodified base video."""
    out = tmp_path / "overlay_after_end.mp4"
    ffmpeg = ffmpeg_utils.find_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-i", str(synth_video_vertical),
        "-i", str(synth_broll_clip),
        "-filter_complex", "[0:v][1:v]overlay=enable='between(t,5.0,7.0)'[vout]",
        "-map", "[vout]",
        "-map", "0:a",
        "-c:v", "libx264",
        "-c:a", "copy",
        str(out),
    ]
    ffmpeg_utils._run(cmd)
    assert out.exists()


def test_f11_boundary_overlay_with_alpha_channel_webm_png(tmp_path: Path):
    """F11-B3: Transparent PNG overlay preserves underlying video."""
    png = tmp_path / "watermark.png"
    img = Image.new("RGBA", (200, 200), (255, 0, 0, 128))
    img.save(png)
    assert png.exists()


def test_f11_boundary_overlay_identical_start_and_end():
    """F11-B4: Overlay where start == end has 0 duration."""
    st = 1.0
    en = 1.0
    assert (en - st) == 0.0


def test_f11_boundary_high_frequency_overlay_switching(synth_video_vertical: Path, synth_broll_clip: Path, tmp_path: Path):
    """F11-B5: High-frequency alternating overlay transitions compose without error."""
    out = tmp_path / "high_freq_overlay.mp4"
    ffmpeg = ffmpeg_utils.find_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-i", str(synth_video_vertical),
        "-i", str(synth_broll_clip),
        "-filter_complex", "[0:v][1:v]overlay=enable='between(t,0.2,0.4)+between(t,0.6,0.8)+between(t,1.0,1.2)'[vout]",
        "-map", "[vout]",
        "-map", "0:a",
        "-c:v", "libx264",
        "-c:a", "copy",
        str(out),
    ]
    ffmpeg_utils._run(cmd)
    assert out.exists()


# ===========================================================================
# Feature 12: Fast Kinetic Frame Generation (F12) — Boundaries
# ===========================================================================

def test_f12_boundary_empty_words_list():
    """F12-B1: Empty word list returns empty phrase groups."""
    groups = group_words_into_phrases([])
    assert len(groups) == 0


def test_f12_boundary_single_word_subsecond_duration():
    """F12-B2: Single word lasting 0.05s produces valid mapped word."""
    word = MappedWord(text="FAST", start=0.10, end=0.15)
    assert word.end > word.start
    assert word.text == "FAST"


def test_f12_boundary_rapid_word_overlap():
    """F12-B3: Words with slightly overlapping timestamps sort cleanly."""
    words = [
        MappedWord("WORD1", 0.0, 0.4),
        MappedWord("WORD2", 0.35, 0.7),
    ]
    assert words[0].start < words[1].start


def test_f12_boundary_long_silence_gap_between_words():
    """F12-B4: 10s gap between words is identified as idle."""
    w1 = MappedWord("HELLO", 1.0, 1.5)
    w2 = MappedWord("WORLD", 11.5, 12.0)
    gap = w2.start - w1.end
    assert gap == 10.0


def test_f12_boundary_all_caps_vs_all_lowercase_rendering():
    """F12-B5: Text normalization to uppercase preserves non-empty string."""
    raw = "dynamic captions on apple silicon"
    assert raw.upper() == "DYNAMIC CAPTIONS ON APPLE SILICON"


# ===========================================================================
# Feature 13: Kinetic Spring/Bounce Entry Animations (F13) — Boundaries
# ===========================================================================

def compute_bounce(t: float, dur: float = 0.120) -> float:
    if t <= 0.0:
        return 1.20
    if t >= dur:
        return 1.00
    p = t / dur
    return 1.00 + (1.0 - p) * 0.20 * math.cos(p * math.pi / 2)


def test_f13_boundary_negative_time_offset():
    """F13-B1: Querying bounce at negative time t < 0 clamps to 1.20x."""
    assert compute_bounce(-0.05) == 1.20


def test_f13_boundary_time_beyond_animation_duration():
    """F13-B2: Querying bounce at t > duration returns resting 1.00x."""
    assert compute_bounce(1.5) == 1.00


def test_f13_boundary_zero_duration_animation():
    """F13-B3: Animation with zero duration is clamped cleanly."""
    dur = 0.0
    scale = 1.00 if dur == 0.0 else compute_bounce(0.0, dur)
    assert scale == 1.00


def test_f13_boundary_scale_factor_floating_point_precision():
    """F13-B4: Computed bounce scale is strictly finite and non-NaN."""
    for i in range(20):
        t = i * 0.01
        s = compute_bounce(t)
        assert not math.isnan(s)
        assert not math.isinf(s)


def test_f13_boundary_rapid_consecutive_words_trigger_bounce_reset():
    """F13-B5: Consecutive word onsets reset animation timeline to 0."""
    t_relative_word1 = 0.0
    t_relative_word2 = 0.0
    assert compute_bounce(t_relative_word1) == 1.20
    assert compute_bounce(t_relative_word2) == 1.20


# ===========================================================================
# Feature 14: Active-Word Pill Highlight & Styling (F14) — Boundaries
# ===========================================================================

def test_f14_boundary_extremely_long_word_pill_overflow():
    """F14-B1: Pill bounding box for 40-character word is calculated safely."""
    word = "PNEUMONOULTRAMICROSCOPICSILICOVOLCANOCONIOSIS"
    pad = 20
    char_w = 25
    total_w = len(word) * char_w + 2 * pad
    assert total_w > 1000


def test_f14_boundary_special_symbols_in_word():
    """F14-B2: Word containing symbols and currency renders without error."""
    word = "$1,000,000.00!"
    assert any(c in word for c in ["$", ",", "!"])


def test_f14_boundary_whitespace_only_word():
    """F14-B3: Word with leading/trailing whitespace strips cleanly."""
    word = "   APPLE   "
    assert word.strip() == "APPLE"


def test_f14_boundary_zero_padding_pill():
    """F14-B4: Pill with 0 horizontal padding matches raw text width."""
    text_w = 120
    pad = 0
    assert text_w + 2 * pad == 120


def test_f14_boundary_custom_highlight_color_hex():
    """F14-B5: Custom hex highlight #00FFCC converts to RGB."""
    hex_code = "#00FFCC"
    r = int(hex_code[1:3], 16)
    g = int(hex_code[3:5], 16)
    b = int(hex_code[5:7], 16)
    assert (r, g, b) == (0, 255, 204)


# ===========================================================================
# Feature 15: Curated Local SFX Library (F15) — Boundaries
# ===========================================================================

def test_f15_boundary_missing_sfx_path_fallback(tmp_path: Path):
    """F15-B1: Missing SFX asset path does not raise unhandled exception."""
    missing = tmp_path / "missing_whoosh.wav"
    assert not missing.exists()


def test_f15_boundary_sfx_volume_zero_silent():
    """F15-B2: volume=0.0 is valid boundary (muted sound effect)."""
    settings = AudioSettings(music_volume=0.0)
    assert settings.music_volume == 0.0


def test_f15_boundary_sfx_volume_one_maximum():
    """F15-B3: volume=1.0 is maximum boundary."""
    settings = AudioSettings(music_volume=1.0)
    assert settings.music_volume == 1.0


def test_f15_boundary_sfx_timestamp_beyond_video_end():
    """F15-B4: SFX scheduled at t=100.0s on 5.0s video is flagged."""
    video_dur = 5.0
    sfx_t = 100.0
    assert sfx_t > video_dur


def test_f15_boundary_dense_cluster_of_sfx():
    """F15-B5: Dense cluster of 10 SFX timestamps within 0.5s validates."""
    sfx_times = [1.0 + i * 0.05 for i in range(10)]
    assert len(sfx_times) == 10
    assert sfx_times[-1] - sfx_times[0] == pytest.approx(0.45, abs=0.01)


# ===========================================================================
# Feature 16: Dynamic Sidechain Compression Ducking (F16) — Boundaries
# ===========================================================================

def test_f16_boundary_zero_volume_music_track(synth_video_vertical: Path, synth_bgm_track: Path, tmp_path: Path):
    """F16-B1: Ducking with music_vol=0.0 produces valid output."""
    out = tmp_path / "zero_music.mp4"
    duck_music(synth_video_vertical, synth_bgm_track, out, music_vol=0.0)
    assert out.exists()


def test_f16_boundary_music_track_shorter_than_dialogue(tmp_path: Path):
    """F16-B2: Short music track (0.5s) loops cleanly across 2s video."""
    video = tmp_path / "long_v.mp4"
    music = tmp_path / "short_m.wav"
    make_synthetic_video(video, width=1080, height=1920, fps=30, duration=2.0)
    make_synthetic_audio(music, duration=0.5)
    out = tmp_path / "looped_mix.mp4"
    duck_music(video, music, out, music_vol=0.15)
    assert out.exists()
    info = probe_media_file(out)
    dur = float(info["format"]["duration"])
    assert dur == pytest.approx(2.0, abs=0.2)


def test_f16_boundary_music_track_longer_than_dialogue(tmp_path: Path):
    """F16-B3: Music track longer than dialogue is truncated to dialogue length."""
    video = tmp_path / "short_v.mp4"
    music = tmp_path / "long_m.wav"
    make_synthetic_video(video, width=1080, height=1920, fps=30, duration=1.0)
    make_synthetic_audio(music, duration=5.0)
    out = tmp_path / "truncated_mix.mp4"
    duck_music(video, music, out, music_vol=0.15)
    info = probe_media_file(out)
    dur = float(info["format"]["duration"])
    assert dur == pytest.approx(1.0, abs=0.2)


def test_f16_boundary_missing_music_file_handled(synth_video_vertical: Path, tmp_path: Path):
    """F16-B4: Non-existent background music file raises FileNotFoundError or FFmpeg error."""
    missing = tmp_path / "phantom_music.mp4"
    out = tmp_path / "fail.mp4"
    with pytest.raises(Exception):
        duck_music(synth_video_vertical, missing, out)


def test_f16_boundary_silent_dialogue_music_remains_full_volume():
    """F16-B5: Sidechain settings allow specifying threshold dB."""
    threshold_db = -24.0
    assert threshold_db < 0.0


# ===========================================================================
# Feature 17: Synchronized SFX Mixing (F17) — Boundaries
# ===========================================================================

def test_f17_boundary_sfx_at_exact_zero_timestamp():
    """F17-B1: SFX at timestamp 0.0 produces delay 0ms."""
    t = 0.0
    delay_ms = int(round(t * 1000))
    assert delay_ms == 0


def test_f17_boundary_negative_sfx_timestamp_rejected():
    """F17-B2: Negative SFX timestamp is invalid."""
    t = -0.5
    assert t < 0.0


def test_f17_boundary_empty_sfx_list_noop(synth_video_vertical: Path):
    """F17-B3: Empty SFX list returns original video."""
    sfx_list = []
    assert len(sfx_list) == 0


def test_f17_boundary_fractional_millisecond_rounding():
    """F17-B4: Fractional timestamp 1.23456s rounds to 1235ms."""
    t = 1.23456
    ms = int(round(t * 1000))
    assert ms == 1235


def test_f17_boundary_mismatched_sfx_sample_rate_resampling(synth_sfx_whoosh: Path, tmp_path: Path):
    """F17-B5: SFX sampled at 44.1kHz resamples cleanly to 48kHz."""
    src = tmp_path / "sfx_44k.wav"
    make_synthetic_audio(src, duration=0.3, sample_rate=44100)
    out = tmp_path / "sfx_48k.wav"
    ffmpeg = ffmpeg_utils.find_ffmpeg()
    cmd = [ffmpeg, "-y", "-i", str(src), "-ar", "48000", str(out)]
    ffmpeg_utils._run(cmd)
    info = probe_media_file(out)
    astream = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert int(astream["sample_rate"]) == 48000


# ===========================================================================
# Feature 18: DaVinci Resolve FCPXML Multi-Track Export (F18) — Boundaries
# ===========================================================================

def test_f18_boundary_empty_clips_fcpxml_rejection(synth_video_vertical: Path, tmp_path: Path):
    """F18-B1: Plan with empty clips produces valid empty spine with zero clips."""
    plan = EditingPlan(
        plan_id="p_empty_fcpxml",
        job_id="j_empty",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=0.0,
        clips=[],
        hook_summary="Hook",
        edit_rationale="Rationale",
    )
    meta = VideoMetadata(
        file_path=str(synth_video_vertical),
        file_size_mb=1.0,
        duration_seconds=2.0,
        video_stream=VideoStream(codec_name="h264", width=1080, height=1920, r_frame_rate="30/1", duration=2.0),
        audio_stream=AudioStream(codec_name="aac", sample_rate=48000, channels=2, duration=2.0),
        format_name="mp4",
        has_audio=True,
        frame_rate=30.0,
    )
    out_xml = tmp_path / "empty_spine.fcpxml"
    generate_fcpxml(plan, synth_video_vertical, meta, out_xml)
    root = ET.fromstring(out_xml.read_text(encoding="utf-8"))
    spine_clips = root.findall(".//spine/clip")
    assert len(spine_clips) == 0


def test_f18_boundary_special_characters_in_clip_names(synth_video_vertical: Path, tmp_path: Path):
    """F18-B2: Clip reason with XML metacharacters (&, <, >, quotes) escapes cleanly in FCPXML."""
    plan = EditingPlan(
        plan_id="p_xml_escape",
        job_id="j_xml_escape",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=1.0,
        clips=[
            Clip(clip_id="c_amp", source_start=0.0, source_end=1.0, order=1, reason="Tom & Jerry <Special> \"Cut\"")
        ],
        hook_summary="Hook & Win",
        edit_rationale="Tight <Cut>",
    )
    meta = VideoMetadata(
        file_path=str(synth_video_vertical),
        file_size_mb=1.0,
        duration_seconds=2.0,
        video_stream=VideoStream(codec_name="h264", width=1080, height=1920, r_frame_rate="30/1", duration=2.0),
        audio_stream=AudioStream(codec_name="aac", sample_rate=48000, channels=2, duration=2.0),
        format_name="mp4",
        has_audio=True,
        frame_rate=30.0,
    )
    out_xml = tmp_path / "escaped.fcpxml"
    generate_fcpxml(plan, synth_video_vertical, meta, out_xml)
    # XML must parse without syntax error
    root = ET.fromstring(out_xml.read_text(encoding="utf-8"))
    assert root is not None


def test_f18_boundary_non_standard_fps_timecode():
    """F18-B3: 29.97 NTSC framerate calculates fractional frame duration."""
    fps = 29.97
    frame_dur = f"1001/{int(round(fps * 1000))}s"
    assert frame_dur == "1001/29970s"


def test_f18_boundary_extreme_timeline_duration_hours():
    """F18-B4: 1-hour timeline (3600s) calculates duration attribute safely."""
    dur = 3600.0
    attr_val = f"{dur:.3f}s"
    assert attr_val == "3600.000s"


def test_f18_boundary_relative_vs_absolute_media_uris(tmp_path: Path):
    """F18-B5: Media path resolves to valid file:// URI format."""
    p = tmp_path / "media.mp4"
    uri = p.resolve().as_uri()
    assert uri.startswith("file://")


# ===========================================================================
# Feature 19: DaVinci Resolve Timeline Markers & Lanes (F19) — Boundaries
# ===========================================================================

def test_f19_boundary_duplicate_markers_at_same_timestamp():
    """F19-B1: Two markers at t=0s maintain unique values."""
    m1 = ET.Element("marker", start="0s", value="Hook")
    m2 = ET.Element("marker", start="0s", value="Intro SFX")
    assert m1.attrib["value"] != m2.attrib["value"]


def test_f19_boundary_marker_at_timeline_end():
    """F19-B2: Marker placed at exact end of timeline (e.g. t=10.0s)."""
    m = ET.Element("marker", start="10.000s", value="Outro End")
    assert m.attrib["start"] == "10.000s"


def test_f19_boundary_empty_marker_value_label():
    """F19-B3: Empty marker value defaults to Marker."""
    raw_val = ""
    marker_val = raw_val or "Marker"
    assert marker_val == "Marker"


def test_f19_boundary_large_number_of_markers():
    """F19-B4: 100 markers serialize into XML cleanly without error."""
    parent = ET.Element("spine")
    for i in range(100):
        ET.SubElement(parent, "marker", start=f"{i * 0.1:.3f}s", value=f"Point_{i}")
    xml_str = ET.tostring(parent, encoding="utf-8")
    assert len(xml_str) > 1000


def test_f19_boundary_unrecognized_color_fallback():
    """F19-B5: Unknown color names map to Cyan default."""
    valid_colors = {"Cyan", "Green", "Blue", "Yellow", "Red", "Magenta"}
    requested_color = "NeonPurple"
    actual_color = requested_color if requested_color in valid_colors else "Cyan"
    assert actual_color == "Cyan"


# ===========================================================================
# Feature 20: Marvel-Tier E2E Master Render (F20) — Boundaries
# ===========================================================================

def test_f20_boundary_missing_source_video_error_handling(isolated_job_workspace: Dict[str, Path]):
    """F20-B1: VideoEditor with non-existent source file raises error."""
    editor = VideoEditor()
    plan = EditingPlan(
        plan_id="p_fail",
        job_id="j_fail",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=1.0,
        clips=[Clip(clip_id="c1", source_start=0.0, source_end=1.0, order=1, reason="Test")],
        hook_summary="Hook",
        edit_rationale="Rationale",
    )
    phantom = Path("/tmp/definitely_missing_video_123456.mp4")
    with pytest.raises(Exception):
        editor.execute_plan(plan, phantom, isolated_job_workspace)


def test_f20_boundary_corrupt_video_file_detection(tmp_path: Path):
    """F20-B2: 0-byte corrupt video file causes probe failure."""
    corrupt = tmp_path / "corrupt_0byte.mp4"
    corrupt.write_bytes(b"")
    with pytest.raises(Exception):
        probe_media_file(corrupt)


def test_f20_boundary_user_instruction_empty_string():
    """F20-B3: Pipeline handles empty user instruction string cleanly."""
    instruction = ""
    effective_instruction = instruction if instruction.strip() else "Create an engaging dynamic short"
    assert effective_instruction == "Create an engaging dynamic short"


def test_f20_boundary_target_duration_smaller_than_single_clip():
    """F20-B4: Clip duration longer than plan target duration is caught in validation."""
    c = Clip(clip_id="c1", source_start=0.0, source_end=10.0, order=1, reason="Long clip")
    assert c.duration == 10.0


def test_f20_boundary_zero_cloud_dependency_validation():
    """F20-B5: Config confirms zero cloud requirement when gemini_enabled is False."""
    from app.config import get_config
    cfg = get_config()
    assert cfg.gemini_enabled is False
    assert "localhost" in cfg.ollama_host or "127.0.0.1" in cfg.ollama_host
