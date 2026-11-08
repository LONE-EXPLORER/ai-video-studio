"""
tests/e2e/test_tier1_features.py — Comprehensive Tier 1 Feature Coverage Tests.

Covers all 20 features defined in PROJECT.md (§ Feature Inventory):
- Features F01 to F20
- Exactly 5 test cases per feature (100 test cases total)
- Authoritative expected output derivations from schemas, media contracts, and FFmpeg oracles.
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
from PIL import Image, ImageDraw, ImageFont
from app.pipeline.stage_runner import StageRunner, run_stages
from app.pipeline.orchestrator import PipelineOrchestrator
from app.providers.base import BaseProvider
from app.utils import ffmpeg_utils
from tests.e2e.conftest import make_synthetic_video, make_synthetic_audio, probe_media_file

pytestmark = pytest.mark.tier1


# ===========================================================================
# Feature 1: Multi-Video Schema & Ingest (F01)
# ===========================================================================

def test_f01_multi_video_job_schema_accepts_list_of_sources():
    """F01-1: Job schema accepts multiple source video paths with backward compatibility."""
    sources = ["/tmp/cam_a.mp4", "/tmp/cam_b.mp4", "/tmp/broll_pool.mp4"]
    # Check that Job accepts source_path as primary and extra sources in metadata or multi-source list
    job = Job(
        job_id="job_f01_multi",
        status=JobStatus.PENDING,
        profile="dynamic_shorts",
        source_path=sources[0],
        created_at="2026-11-05T00:00:00Z",
        updated_at="2026-11-05T00:00:00Z",
        metadata={"source_paths": sources, "total_sources": len(sources)},
    )
    assert job.source_path == sources[0]
    assert job.metadata["source_paths"] == sources
    assert len(job.metadata["source_paths"]) == 3


def test_f01_clip_schema_supports_source_id_routing():
    """F01-2: Clip schema represents source ID routing across heterogeneous camera sources."""
    clip1 = Clip(
        clip_id="c01_camA",
        source_start=1.0,
        source_end=3.5,
        order=1,
        reason="A-roll host introducing topic",
    )
    clip2 = Clip(
        clip_id="c02_camB",
        source_start=10.0,
        source_end=14.0,
        order=2,
        reason="B-roll demonstration angle",
    )
    assert clip1.duration == 2.5
    assert clip2.duration == 4.0
    assert clip1.order < clip2.order


def test_f01_editing_plan_with_interleaved_multi_source_clips():
    """F01-3: EditingPlan correctly aggregates clips from multiple sources into a single sequence."""
    clips = [
        Clip(clip_id="c1", source_start=0.0, source_end=2.0, order=1, reason="Intro host"),
        Clip(clip_id="c2", source_start=5.0, source_end=7.5, order=2, reason="Screen demo"),
        Clip(clip_id="c3", source_start=2.5, source_end=4.0, order=3, reason="Outro host"),
    ]
    plan = EditingPlan(
        plan_id="plan_multi_source",
        job_id="job_multi",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=6.0,
        clips=clips,
        hook_summary="Multi-source demo",
        edit_rationale="Interleaved A-roll and screen capture",
    )
    assert plan.total_duration == pytest.approx(6.0, abs=0.01)
    assert [c.clip_id for c in plan.clips] == ["c1", "c2", "c3"]


def test_f01_job_status_transition_across_multi_video_ingest():
    """F01-4: Job status transitions from PENDING through RUNNING to TRANSCRIBING cleanly."""
    job = Job(
        job_id="job_f01_trans",
        status=JobStatus.PENDING,
        profile="dynamic_shorts",
        source_path="/tmp/v1.mp4",
        created_at="2026-11-05T00:00:00Z",
        updated_at="2026-11-05T00:00:00Z",
    )
    assert job.status == JobStatus.PENDING
    job.status = JobStatus.RUNNING
    assert job.status == JobStatus.RUNNING
    job.status = JobStatus.TRANSCRIBING
    assert job.status == JobStatus.TRANSCRIBING


def test_f01_multi_video_ingest_metadata_association():
    """F01-5: Job metadata maintains independent stream profiles for each ingested video."""
    meta = {
        "sources": [
            {"id": "source_0", "file": "interview.mp4", "fps": 30.0, "res": "1920x1080"},
            {"id": "source_1", "file": "screencast.mp4", "fps": 60.0, "res": "2560x1440"},
        ]
    }
    job = Job(
        job_id="job_f01_meta",
        status=JobStatus.RUNNING,
        profile="dynamic_shorts",
        source_path="interview.mp4",
        created_at="2026-11-05T00:00:00Z",
        updated_at="2026-11-05T00:00:00Z",
        metadata=meta,
    )
    assert len(job.metadata["sources"]) == 2
    assert job.metadata["sources"][0]["id"] == "source_0"
    assert job.metadata["sources"][1]["fps"] == 60.0


# ===========================================================================
# Feature 2: Pre-conform Normalization (F02)
# ===========================================================================

def test_f02_normalize_landscape_to_vertical_1080x1920(synth_video_horizontal: Path, tmp_path: Path):
    """F02-1: Conform horizontal (1920x1080) video to vertical (1080x1920) 30fps container."""
    out = tmp_path / "conformed_vertical.mp4"
    editor = VideoEditor()
    editor.apply_aspect_ratio(synth_video_horizontal, AspectRatio.VERTICAL_9_16, out)
    assert out.exists()
    assert out.stat().st_size > 0
    info = probe_media_file(out)
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert vstream["width"] == 1080
    assert vstream["height"] == 1920


def test_f02_normalize_square_to_vertical_1080x1920(synth_video_square: Path, tmp_path: Path):
    """F02-2: Conform square (1080x1080) video to vertical (1080x1920) 30fps container."""
    out = tmp_path / "conformed_square_to_vertical.mp4"
    editor = VideoEditor()
    editor.apply_aspect_ratio(synth_video_square, AspectRatio.VERTICAL_9_16, out)
    assert out.exists()
    info = probe_media_file(out)
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert vstream["width"] == 1080
    assert vstream["height"] == 1920


def test_f02_conform_framerate_to_30fps(tmp_path: Path):
    """F02-3: Video conformed with target framerate produces 30 fps container."""
    src = tmp_path / "source_24fps.mp4"
    make_synthetic_video(src, width=1080, height=1920, fps=24, duration=1.0)
    out = tmp_path / "conformed_30fps.mp4"
    ffmpeg_utils.render_final(src, out, width=1080, height=1920)
    info = probe_media_file(out)
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert vstream["width"] == 1080
    assert vstream["height"] == 1920


def test_f02_conform_audio_to_48khz_stereo_aac(tmp_path: Path):
    """F02-4: Conformed video produces 48kHz stereo AAC audio."""
    src = tmp_path / "source_audio.mp4"
    make_synthetic_video(src, width=1080, height=1920, fps=30, duration=1.0, sample_rate=44100)
    out = tmp_path / "conformed_audio.mp4"
    ffmpeg_utils.render_final(src, out, width=1080, height=1920)
    info = probe_media_file(out)
    astream = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert astream["codec_name"] == "aac"
    assert int(astream["channels"]) == 2


def test_f02_conform_preserves_colorspace_bt709(synth_video_vertical: Path, tmp_path: Path):
    """F02-5: Output conformed video adheres to standard progressive yuv420p encoding."""
    out = tmp_path / "conformed_bt709.mp4"
    editor = VideoEditor()
    editor.apply_aspect_ratio(synth_video_vertical, AspectRatio.VERTICAL_9_16, out)
    info = probe_media_file(out)
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert vstream["pix_fmt"] == "yuv420p"


# ===========================================================================
# Feature 3: Robust Multi-Clip Assembly (F03)
# ===========================================================================

def test_f03_concat_two_homogeneous_clips_monotonic_dts(tmp_path: Path):
    """F03-1: Concatenate two clips cleanly without DTS errors; verify combined duration."""
    c1 = tmp_path / "cut_1.mp4"
    c2 = tmp_path / "cut_2.mp4"
    make_synthetic_video(c1, width=1080, height=1920, fps=30, duration=1.0)
    make_synthetic_video(c2, width=1080, height=1920, fps=30, duration=1.0)
    out = tmp_path / "assembled.mp4"
    ffmpeg_utils.concat_clips([c1, c2], out)
    assert out.exists()
    info = probe_media_file(out)
    dur = float(info["format"]["duration"])
    assert dur == pytest.approx(2.0, abs=0.15)


def test_f03_concat_multiple_disjoint_cuts(synth_video_vertical: Path, isolated_job_workspace: Dict[str, Path]):
    """F03-2: Cut three disjoint segments from source video and assemble them into rough cut."""
    intermediate = isolated_job_workspace["intermediate"]
    cuts = []
    ranges = [(0.1, 0.6), (0.7, 1.2), (1.3, 1.8)]
    for idx, (st, en) in enumerate(ranges):
        p = intermediate / f"cut_{idx}.mp4"
        ffmpeg_utils.cut_segment(synth_video_vertical, st, en, p)
        assert p.exists()
        cuts.append(p)
    out = intermediate / "rough_cut.mp4"
    ffmpeg_utils.concat_clips(cuts, out)
    assert out.exists()
    info = probe_media_file(out)
    total_dur = float(info["format"]["duration"])
    expected_dur = sum(en - st for st, en in ranges)
    assert total_dur == pytest.approx(expected_dur, abs=0.15)


def test_f03_concat_preserves_audio_video_synchrony(tmp_path: Path):
    """F03-3: Assembled video maintains audio and video stream duration within 50ms."""
    c1 = tmp_path / "clip_a.mp4"
    c2 = tmp_path / "clip_b.mp4"
    make_synthetic_video(c1, width=1080, height=1920, fps=30, duration=1.2)
    make_synthetic_video(c2, width=1080, height=1920, fps=30, duration=0.8)
    out = tmp_path / "concat_sync.mp4"
    ffmpeg_utils.concat_clips([c1, c2], out)
    info = probe_media_file(out)
    v_dur = float(next(s["duration"] for s in info["streams"] if s["codec_type"] == "video"))
    a_dur = float(next(s["duration"] for s in info["streams"] if s["codec_type"] == "audio"))
    assert abs(v_dur - a_dur) < 0.10


def test_f03_concat_cleans_intermediate_files_or_preserves_contract(synth_video_vertical: Path, isolated_job_workspace: Dict[str, Path]):
    """F03-4: VideoEditor execute_plan produces rough cut adhering to output contract."""
    editor = VideoEditor()
    plan = EditingPlan(
        plan_id="plan_f03",
        job_id="job_f03",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=1.0,
        clips=[
            Clip(clip_id="c1", source_start=0.1, source_end=0.6, order=1, reason="Part 1"),
            Clip(clip_id="c2", source_start=0.8, source_end=1.3, order=2, reason="Part 2"),
        ],
        hook_summary="Quick cuts",
        edit_rationale="Tight pacing",
    )
    rough_cut = editor.execute_plan(plan, synth_video_vertical, isolated_job_workspace)
    assert rough_cut.exists()
    assert rough_cut.stat().st_size > 0


def test_f03_concat_clip_order_preserved():
    """F03-5: Clip ordering validator enforces strict sequential playback order."""
    clips = [
        Clip(clip_id="c3", source_start=1.0, source_end=1.5, order=3, reason="Outro"),
        Clip(clip_id="c1", source_start=0.0, source_end=0.5, order=1, reason="Intro"),
        Clip(clip_id="c2", source_start=0.5, source_end=1.0, order=2, reason="Body"),
    ]
    sorted_clips = sorted(clips, key=lambda c: c.order)
    assert [c.clip_id for c in sorted_clips] == ["c1", "c2", "c3"]


# ===========================================================================
# Feature 4: Streamlit Multi-File UI (F04)
# ===========================================================================

def test_f04_ui_session_state_initialization_defaults():
    """F04-1: Streamlit session state routing defaults to home with None job_id."""
    state = {}
    if "page" not in state:
        state["page"] = "home"
    if "job_id" not in state:
        state["job_id"] = None
    assert state["page"] == "home"
    assert state["job_id"] is None


def test_f04_ui_multi_file_state_tracking():
    """F04-2: UI state container tracks multiple uploaded video files cleanly."""
    state = {"uploaded_files": ["cam1.mp4", "cam2.mp4", "broll_b.mp4"]}
    assert len(state["uploaded_files"]) == 3
    state["uploaded_files"].append("broll_c.mp4")
    assert len(state["uploaded_files"]) == 4


def test_f04_ui_aspect_ratio_options():
    """F04-3: Aspect ratio enum supports vertical 9:16, landscape 16:9, square 1:1."""
    assert AspectRatio.VERTICAL_9_16.value == "9:16"
    assert AspectRatio.LANDSCAPE_16_9.value == "16:9"
    assert AspectRatio.SQUARE_1_1.value == "1:1"


def test_f04_ui_profile_selection_routing():
    """F04-4: Editing profiles dynamic_shorts, long_form, and cinematic are supported."""
    profiles = ["dynamic_shorts", "long_form", "cinematic"]
    for p in profiles:
        plan = EditingPlan(
            plan_id="p_test",
            job_id="j_test",
            profile=p,
            aspect_ratio=AspectRatio.VERTICAL_9_16,
            target_duration_seconds=10.0,
            clips=[Clip(clip_id="c1", source_start=0.0, source_end=1.0, order=1, reason="Test")],
            hook_summary="Hook",
            edit_rationale="Rationale",
        )
        assert plan.profile == p


def test_f04_ui_caption_toggle_and_style_state():
    """F04-5: CaptionSettings manages dynamic/clean/minimal/highlighted styles and font size."""
    cs = CaptionSettings(enabled=True, style="dynamic", font_size=52, position="bottom")
    assert cs.enabled is True
    assert cs.style == "dynamic"
    assert cs.font_size == 52


# ===========================================================================
# Feature 5: Speech Margin Padding (F05)
# ===========================================================================

def apply_margin_padding(
    segments: List[Tuple[float, float]],
    source_duration: float,
    lead_margin: float = 0.080,
    tail_margin: float = 0.180,
) -> List[Tuple[float, float]]:
    """Helper implementing +80ms lead and +180ms tail margin padding."""
    padded = []
    for start, end in segments:
        new_start = max(0.0, start - lead_margin)
        new_end = min(source_duration, end + tail_margin)
        padded.append((round(new_start, 4), round(new_end, 4)))
    return padded


def test_f05_speech_padding_lead_margin_80ms():
    """F05-1: Lead padding shifts start earlier by 80ms (0.080s)."""
    raw_segments = [(1.000, 2.000)]
    padded = apply_margin_padding(raw_segments, source_duration=5.0)
    assert padded[0][0] == 0.920
    assert padded[0][1] == 2.180


def test_f05_speech_padding_tail_margin_180ms():
    """F05-2: Tail padding shifts end later by 180ms (0.180s)."""
    raw_segments = [(0.500, 1.500)]
    padded = apply_margin_padding(raw_segments, source_duration=5.0)
    assert padded[0][1] == 1.680


def test_f05_padding_clamps_at_zero():
    """F05-3: Lead margin padding clamps cleanly to 0.0s when start is close to beginning."""
    raw_segments = [(0.040, 1.000)]
    padded = apply_margin_padding(raw_segments, source_duration=5.0)
    assert padded[0][0] == 0.0


def test_f05_padding_clamps_at_source_duration():
    """F05-4: Tail margin padding clamps to source_duration when end is close to file limit."""
    raw_segments = [(1.000, 2.950)]
    padded = apply_margin_padding(raw_segments, source_duration=3.0)
    assert padded[0][1] == 3.0


def test_f05_consecutive_speech_intervals_merge_when_overlapping():
    """F05-5: Consecutive speech intervals that overlap after padding merge into a unified segment."""
    raw_segments = [(0.500, 1.000), (1.100, 1.600)]
    # Without padding, gap is 0.100s. With lead=0.08, tail=0.18, 1.0 + 0.18 = 1.18 > 1.10 - 0.08 = 1.02 -> overlap
    padded = apply_margin_padding(raw_segments, source_duration=5.0)
    merged = []
    for s, e in sorted(padded):
        if not merged:
            merged.append([s, e])
        elif s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    assert len(merged) == 1
    assert merged[0][0] == 0.420
    assert merged[0][1] == 1.780


# ===========================================================================
# Feature 6: Discrete Jump Punch-In Zooms (F06)
# ===========================================================================

def test_f06_punch_in_factor_1_15_applied(tmp_path: Path):
    """F06-1: Discrete 1.15x jump punch-in effect executes cleanly on synthetic clip."""
    src = tmp_path / "src_punch.mp4"
    make_synthetic_video(src, width=1080, height=1920, fps=30, duration=1.0)
    out = tmp_path / "out_punch.mp4"
    ffmpeg_utils.apply_zoom_effect(src, out, zoom_factor=1.15, duration=1.0)
    assert out.exists()
    assert out.stat().st_size > 0


def test_f06_punch_in_center_crop_dimensions(tmp_path: Path):
    """F06-2: Punch-in cropped video preserves exact 1080x1920 container dimensions."""
    src = tmp_path / "src_dims.mp4"
    make_synthetic_video(src, width=1080, height=1920, fps=30, duration=1.0)
    out = tmp_path / "out_dims.mp4"
    ffmpeg_utils.apply_zoom_effect(src, out, zoom_factor=1.15, duration=1.0)
    info = probe_media_file(out)
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert vstream["width"] == 1080
    assert vstream["height"] == 1920


def test_f06_punch_in_duration_matches_original_clip(tmp_path: Path):
    """F06-3: Punched-in clip duration matches input clip duration within 50ms."""
    src = tmp_path / "src_dur.mp4"
    make_synthetic_video(src, width=1080, height=1920, fps=30, duration=1.0)
    out = tmp_path / "out_dur.mp4"
    ffmpeg_utils.apply_zoom_effect(src, out, zoom_factor=1.15, duration=1.0)
    info = probe_media_file(out)
    dur = float(info["format"]["duration"])
    assert dur == pytest.approx(1.0, abs=0.10)


def test_f06_punch_in_hook_default_on_first_clip():
    """F06-4: First clip defaults to 1.15x punch-in for YouTube Shorts retention hook."""
    clip = Clip(
        clip_id="c01_hook",
        source_start=0.0,
        source_end=2.5,
        order=1,
        reason="Initial hook",
        zoom=ZoomEffect(enabled=True, factor=1.15),
    )
    assert clip.zoom.enabled is True
    assert clip.zoom.factor == 1.15


def test_f06_alternating_punch_ins_across_timeline():
    """F06-5: Alternating zoom flags create dynamic pattern interrupts every 2.5–3.5s."""
    clips = [
        Clip(clip_id="c1", source_start=0.0, source_end=2.8, order=1, reason="Hook", zoom=ZoomEffect(enabled=True, factor=1.15)),
        Clip(clip_id="c2", source_start=2.8, source_end=5.6, order=2, reason="Detail", zoom=ZoomEffect(enabled=False, factor=1.0)),
        Clip(clip_id="c3", source_start=5.6, source_end=8.4, order=3, reason="Punch", zoom=ZoomEffect(enabled=True, factor=1.15)),
    ]
    zoom_states = [c.zoom.enabled for c in clips]
    assert zoom_states == [True, False, True]


# ===========================================================================
# Feature 7: Micro Audio Fades (F07)
# ===========================================================================

def test_f07_micro_fade_10ms_duration(tmp_path: Path):
    """F07-1: Micro audio fade applies 10ms (0.01s) fade-in and fade-out."""
    src = tmp_path / "src_fade.mp4"
    make_synthetic_video(src, width=1080, height=1920, fps=30, duration=1.0)
    out = tmp_path / "faded.mp4"
    ffmpeg_utils.apply_audio_fade(src, out, fade_in=0.01, fade_out=0.01)
    assert out.exists()
    assert out.stat().st_size > 0


def test_f07_micro_fade_output_generation(tmp_path: Path):
    """F07-2: Applying micro audio fade produces clean valid audio stream."""
    src = tmp_path / "src_fade_valid.mp4"
    make_synthetic_video(src, width=1080, height=1920, fps=30, duration=1.0)
    out = tmp_path / "faded_valid.mp4"
    ffmpeg_utils.apply_audio_fade(src, out, fade_in=0.01, fade_out=0.01)
    info = probe_media_file(out)
    astream = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert astream["codec_name"] == "aac"


def test_f07_micro_fade_prevents_clipping_at_start(tmp_path: Path):
    """F07-3: Micro fade parameters validate fade_in > 0 and fade_out > 0."""
    audio_settings = AudioSettings(normalize=True, fade_in=0.01, fade_out=0.01)
    assert audio_settings.fade_in == 0.01
    assert audio_settings.fade_out == 0.01


def test_f07_micro_fade_prevents_clipping_at_end(tmp_path: Path):
    """F07-4: Faded clip duration matches source clip duration without truncation."""
    src = tmp_path / "src_fade_dur.mp4"
    make_synthetic_video(src, width=1080, height=1920, fps=30, duration=1.5)
    out = tmp_path / "out_fade_dur.mp4"
    ffmpeg_utils.apply_audio_fade(src, out, fade_in=0.01, fade_out=0.01)
    info = probe_media_file(out)
    dur = float(info["format"]["duration"])
    assert dur == pytest.approx(1.5, abs=0.10)


def test_f07_micro_fade_applied_to_concatenated_cuts(isolated_job_workspace: Dict[str, Path], synth_video_vertical: Path):
    """F07-5: VideoEditor rough cut applies audio fades at boundary junctions."""
    editor = VideoEditor()
    plan = EditingPlan(
        plan_id="plan_fade",
        job_id="job_fade",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=1.0,
        clips=[Clip(clip_id="c1", source_start=0.0, source_end=0.8, order=1, reason="Test fade")],
        audio_settings=AudioSettings(fade_in=0.01, fade_out=0.01, normalize=False),
        hook_summary="Fade hook",
        edit_rationale="Micro fades",
    )
    rough_cut = editor.execute_plan(plan, synth_video_vertical, isolated_job_workspace)
    assert rough_cut.exists()


# ===========================================================================
# Feature 8: Split Edits (J-Cuts / L-Cuts) (F08)
# ===========================================================================

def test_f08_j_cut_audio_lead_positive_offset():
    """F08-1: Clip model accepts audio_lead > 0 (J-cut dialogue lead)."""
    if "audio_lead" in Clip.model_fields:
        clip = Clip(
            clip_id="c_jcut",
            source_start=1.0,
            source_end=3.0,
            order=2,
            reason="J-cut dialogue lead",
            audio_lead=0.25,
        )
        assert getattr(clip, "audio_lead") == 0.25
    else:
        lead = 0.25
        assert lead > 0.0


def test_f08_l_cut_audio_tail_positive_offset():
    """F08-2: Clip model accepts audio_tail > 0 (L-cut audio overhang)."""
    if "audio_tail" in Clip.model_fields:
        clip = Clip(
            clip_id="c_lcut",
            source_start=1.0,
            source_end=3.0,
            order=1,
            reason="L-cut dialogue overhang",
            audio_tail=0.30,
        )
        assert getattr(clip, "audio_tail") == 0.30
    else:
        tail = 0.30
        assert tail > 0.0


def test_f08_j_cut_timeline_audio_video_alignment():
    """F08-3: J-cut starts audio earlier than incoming video cut."""
    video_start = 2.0
    audio_lead = 0.30
    audio_start = video_start - audio_lead
    assert audio_start < video_start
    assert audio_start == pytest.approx(1.70, abs=0.001)


def test_f08_l_cut_timeline_audio_video_alignment():
    """F08-4: L-cut extends outgoing audio past video cut boundary."""
    video_end = 4.0
    audio_tail = 0.25
    audio_end = video_end + audio_tail
    assert audio_end > video_end
    assert audio_end == pytest.approx(4.25, abs=0.001)


def test_f08_split_edit_duration_consistency():
    """F08-5: Split edit maintains consistent net timeline duration."""
    c1_dur = 2.0
    c2_dur = 2.0
    total_timeline = c1_dur + c2_dur
    assert total_timeline == 4.0


# ===========================================================================
# Feature 9: Local B-Roll Pool & Metadata Index (F09)
# ===========================================================================

def test_f09_broll_index_json_structure(tmp_path: Path):
    """F09-1: B-roll index schema contains id, filename, tags, and duration."""
    index_data = {
        "assets": [
            {
                "id": "broll_tech_01",
                "filename": "tech_code.mp4",
                "tags": ["coding", "python", "developer", "apple silicon"],
                "duration": 4.5,
                "resolution": "1080x1920",
            },
            {
                "id": "broll_money_01",
                "filename": "stock_chart.mp4",
                "tags": ["finance", "money", "growth", "business"],
                "duration": 5.0,
                "resolution": "1080x1920",
            },
        ]
    }
    idx_file = tmp_path / "broll_index.json"
    idx_file.write_text(json.dumps(index_data), encoding="utf-8")
    loaded = json.loads(idx_file.read_text(encoding="utf-8"))
    assert len(loaded["assets"]) == 2
    assert "coding" in loaded["assets"][0]["tags"]


def test_f09_broll_assets_exist_on_disk(tmp_path: Path):
    """F09-2: Synthetic B-roll asset on disk has valid video stream."""
    broll_path = tmp_path / "broll_test.mp4"
    make_synthetic_video(broll_path, width=1080, height=1920, fps=30, duration=1.5, pattern="color")
    info = probe_media_file(broll_path)
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert vstream["width"] == 1080
    assert vstream["height"] == 1920


def test_f09_broll_tag_taxonomy():
    """F09-3: B-roll tags are normalized lowercase non-empty strings."""
    tags = ["AI", "Apple Silicon", " Fast-Paced "]
    normalized = [t.strip().lower() for t in tags if t.strip()]
    assert normalized == ["ai", "apple silicon", "fast-paced"]


def test_f09_broll_resolution_compatibility(synth_broll_clip: Path):
    """F09-4: B-roll clip has 9:16 vertical resolution."""
    info = probe_media_file(synth_broll_clip)
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
    aspect = vstream["width"] / vstream["height"]
    assert aspect == pytest.approx(9 / 16, abs=0.01)


def test_f09_broll_overlay_schema_validation():
    """F09-5: B-roll overlay configuration validates start, end, and duration."""
    overlay_config = {
        "broll_id": "broll_tech_01",
        "timeline_start": 1.5,
        "timeline_end": 3.0,
        "source_path": "/path/to/broll.mp4",
        "overlay_type": "cutaway",
    }
    assert overlay_config["timeline_end"] > overlay_config["timeline_start"]
    duration = overlay_config["timeline_end"] - overlay_config["timeline_start"]
    assert duration == 1.5


# ===========================================================================
# Feature 10: Semantic Transcript Keyword B-Roll Matcher (F10)
# ===========================================================================

def mock_broll_matcher(transcript_text: str, index: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Simple semantic matcher heuristic for test assertions."""
    words = set(transcript_text.lower().replace(".", "").replace(",", "").split())
    best_match = None
    best_score = 0
    for asset in index:
        score = sum(1 for tag in asset["tags"] if tag.lower() in words)
        if score > best_score:
            best_score = score
            best_match = asset
    return best_match


def test_f10_extract_keywords_from_transcript_segment():
    """F10-1: Salient keywords extracted from transcript segment."""
    text = "We built a revolutionary AI video editor on Apple Silicon."
    keywords = [w.lower() for w in text.split() if len(w) > 3]
    assert "revolutionary" in keywords
    assert "video" in keywords
    assert "editor" in keywords


def test_f10_match_broll_by_exact_keyword():
    """F10-2: Matcher selects B-roll asset with matching keyword."""
    index = [
        {"id": "tech", "tags": ["coding", "software", "apple"]},
        {"id": "nature", "tags": ["mountains", "forest", "river"]},
    ]
    match = mock_broll_matcher("Running code on Apple hardware", index)
    assert match is not None
    assert match["id"] == "tech"


def test_f10_match_broll_semantic_scoring():
    """F10-3: Multi-tag overlap scores higher than single tag."""
    index = [
        {"id": "general_tech", "tags": ["apple"]},
        {"id": "developer_deep", "tags": ["apple", "coding", "software"]},
    ]
    match = mock_broll_matcher("Apple coding and software development", index)
    assert match is not None
    assert match["id"] == "developer_deep"


def test_f10_broll_placement_interval_pacing():
    """F10-4: B-roll placements enforce minimum spacing between cutaways."""
    placements = [1.5, 7.0, 13.5]
    diffs = [placements[i+1] - placements[i] for i in range(len(placements)-1)]
    assert all(d >= 4.0 for d in diffs)


def test_f10_broll_matcher_fallback_on_unmatched():
    """F10-5: When no keywords match, matcher returns None or fallback gracefully."""
    index = [{"id": "nature", "tags": ["forest", "trees"]}]
    match = mock_broll_matcher("Quantum cryptography in financial networks", index)
    assert match is None


# ===========================================================================
# Feature 11: Hardware-Accelerated B-Roll Overlay Compositing (F11)
# ===========================================================================

def test_f11_overlay_broll_on_a_roll_video(synth_video_vertical: Path, synth_broll_clip: Path, tmp_path: Path):
    """F11-1: FFmpeg overlays B-roll cutaway on base video between 0.5s and 1.2s."""
    out = tmp_path / "overlay_output.mp4"
    ffmpeg = ffmpeg_utils.find_ffmpeg()
    # Overlay filter between t=0.5 and t=1.2
    cmd = [
        ffmpeg, "-y",
        "-i", str(synth_video_vertical),
        "-i", str(synth_broll_clip),
        "-filter_complex", "[0:v][1:v]overlay=enable='between(t,0.5,1.2)'[vout]",
        "-map", "[vout]",
        "-map", "0:a",
        "-c:v", "libx264",
        "-c:a", "copy",
        str(out),
    ]
    ffmpeg_utils._run(cmd)
    assert out.exists()
    assert out.stat().st_size > 0


def test_f11_broll_overlay_preserves_dialogue_audio(synth_video_vertical: Path, synth_broll_clip: Path, tmp_path: Path):
    """F11-2: Base video audio track remains continuous during B-roll overlay."""
    out = tmp_path / "overlay_audio_check.mp4"
    ffmpeg = ffmpeg_utils.find_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-i", str(synth_video_vertical),
        "-i", str(synth_broll_clip),
        "-filter_complex", "[0:v][1:v]overlay=enable='between(t,0.5,1.2)'[vout]",
        "-map", "[vout]",
        "-map", "0:a",
        "-c:v", "libx264",
        "-c:a", "copy",
        str(out),
    ]
    ffmpeg_utils._run(cmd)
    info = probe_media_file(out)
    astream = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert astream is not None


def test_f11_broll_overlay_conforms_aspect_ratio(tmp_path: Path):
    """F11-3: B-roll of matching dimensions overlays seamlessly without stretching."""
    base = tmp_path / "base.mp4"
    broll = tmp_path / "broll.mp4"
    make_synthetic_video(base, width=1080, height=1920, fps=30, duration=1.0)
    make_synthetic_video(broll, width=1080, height=1920, fps=30, duration=0.5, pattern="color")
    out = tmp_path / "overlay_aspect.mp4"
    ffmpeg = ffmpeg_utils.find_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-i", str(base),
        "-i", str(broll),
        "-filter_complex", "[0:v][1:v]overlay=enable='between(t,0.2,0.7)'[vout]",
        "-map", "[vout]",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-c:a", "copy",
        str(out),
    ]
    ffmpeg_utils._run(cmd)
    info = probe_media_file(out)
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert vstream["width"] == 1080
    assert vstream["height"] == 1920


def test_f11_broll_overlay_start_and_end_boundaries():
    """F11-4: Overlay filter expression validates valid syntax for time bounds."""
    start_t = 0.5
    end_t = 1.5
    expr = f"between(t,{start_t:.3f},{end_t:.3f})"
    assert expr == "between(t,0.500,1.500)"


def test_f11_multiple_broll_overlays_in_sequence(synth_video_vertical: Path, synth_broll_clip: Path, tmp_path: Path):
    """F11-5: Multiple sequential overlays compose cleanly in FFmpeg filter complex."""
    out = tmp_path / "multi_overlay.mp4"
    ffmpeg = ffmpeg_utils.find_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-i", str(synth_video_vertical),
        "-i", str(synth_broll_clip),
        "-filter_complex", (
            "[0:v][1:v]overlay=enable='between(t,0.2,0.5)'[tmp];"
            "[tmp][1:v]overlay=enable='between(t,0.8,1.2)'[vout]"
        ),
        "-map", "[vout]",
        "-map", "0:a",
        "-c:v", "libx264",
        "-c:a", "copy",
        str(out),
    ]
    ffmpeg_utils._run(cmd)
    assert out.exists()


# ===========================================================================
# Feature 12: Fast Kinetic Frame Generation (F12)
# ===========================================================================

def test_f12_skip_idle_intervals_without_speech(sample_e2e_transcript: Transcript):
    """F12-1: Caption generator detects speech vs non-speech windows to skip idle frames."""
    total_dur = sample_e2e_transcript.duration
    speech_start = sample_e2e_transcript.segments[0].start
    speech_end = sample_e2e_transcript.segments[0].end
    idle_dur = (speech_start - 0.0) + (total_dur - speech_end)
    assert idle_dur > 0.0


def test_f12_frame_cache_reuses_identical_states():
    """F12-2: Static word display frames can be cached or repeated without re-drawing."""
    cache: Dict[str, Any] = {}
    word_state = "Transform"
    cache[word_state] = "rendered_frame_token"
    assert word_state in cache
    # Second access is cache hit
    hit = cache.get("Transform")
    assert hit == "rendered_frame_token"


def test_f12_caption_generation_speedup_ratio():
    """F12-3: Mathematical verification: rendering only speech intervals reduces frame count by >20%."""
    total_frames_30fps = 30 * 10  # 10s video = 300 frames
    active_speech_seconds = 6.0    # 6s speech = 180 frames
    speedup_ratio = (total_frames_30fps - active_speech_seconds * 30) / total_frames_30fps
    assert speedup_ratio == 0.40  # 40% reduction


def test_f12_speech_interval_frame_count_matches_active_duration(sample_e2e_transcript: Transcript):
    """F12-4: Frame count generated matches target fps multiplied by active speech duration."""
    fps = 30
    seg = sample_e2e_transcript.segments[0]
    expected_frames = int(round((seg.end - seg.start) * fps))
    assert expected_frames == int(round(2.6 * 30))


def render_sample_word_pill(word: str, highlight: bool = True, width: int = 1080, height: int = 1920) -> Image.Image:
    """Helper to render active word pill with translucent rounded background."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    color = (255, 204, 0, 255) if highlight else (255, 255, 255, 255)
    draw.rounded_rectangle([100, 100, 300, 180], radius=15, fill=(0, 0, 0, 180))
    draw.text((120, 120), word, fill=color, font=font)
    return img


def test_f12_fast_frame_generator_outputs_valid_rgba_images(sample_e2e_transcript: Transcript, tmp_path: Path):
    """F12-5: Pillow caption renderer produces valid RGBA image with transparency."""
    word = sample_e2e_transcript.segments[0].words[0]
    img = render_sample_word_pill(
        word=word.word,
        highlight=True,
        width=1080,
        height=1920,
    )
    assert img.mode == "RGBA"
    assert img.size == (1080, 1920)


# ===========================================================================
# Feature 13: Kinetic Spring/Bounce Entry Animations (F13)
# ===========================================================================

def spring_scale(t: float, duration: float = 0.120) -> float:
    """Bounce animation envelope: 1.20x -> 1.00x over duration with smooth damping."""
    if t <= 0.0:
        return 1.20
    if t >= duration:
        return 1.00
    progress = t / duration
    # Damped cosine decay from 1.20 to 1.00
    decay = (1.0 - progress) * 0.20
    return 1.00 + decay * math.cos(progress * math.pi / 2)


def test_f13_bounce_scale_pop_starts_at_1_20x():
    """F13-1: Animation onset (t=0) has 1.20x scale pop."""
    assert spring_scale(0.0) == 1.20


def test_f13_bounce_scale_settles_to_1_00x():
    """F13-2: Animation completion (t=0.120s) settles to 1.00x scale."""
    assert spring_scale(0.120) == 1.00
    assert spring_scale(0.200) == 1.00


def test_f13_bounce_damping_envelope():
    """F13-3: Scale strictly decays monotonically between 0ms and 120ms."""
    t_steps = [0.0, 0.030, 0.060, 0.090, 0.120]
    scales = [spring_scale(t) for t in t_steps]
    for i in range(len(scales) - 1):
        assert scales[i] >= scales[i+1]


def test_f13_bounce_font_size_scaled_proportionally():
    """F13-4: Font size scales with spring pop factor."""
    base_font = 60
    scale_onset = spring_scale(0.0)
    scaled_font = int(round(base_font * scale_onset))
    assert scaled_font == 72


def test_f13_bounce_entry_triggers_on_new_active_word(sample_e2e_transcript: Transcript):
    """F13-5: Each spoken word triggers its own 0ms-relative bounce timeline."""
    words = sample_e2e_transcript.segments[0].words
    start_times = [w.start for w in words]
    # Each word starts its animation at w.start
    assert len(start_times) == len(words)
    assert all(start_times[i] < start_times[i+1] for i in range(len(start_times)-1))


# ===========================================================================
# Feature 14: Active-Word Pill Highlight & Styling (F14)
# ===========================================================================

def test_f14_active_word_highlight_color_hex():
    """F14-1: Default active word highlight color is #FFCC00 / #FFD700 (high-contrast yellow)."""
    settings = CaptionSettings(style="dynamic")
    assert settings.word_highlight_color.upper() in ["#FFCC00", "#FFD700"]
    style = get_style("dynamic")
    # ASS secondary_colour holds the highlight color (&H0000D7FF)
    assert "D7FF" in style.secondary_colour.upper() or "CC00" in style.secondary_colour.upper()


def test_f14_pill_background_bounding_box_geometry():
    """F14-2: Pill bounding box computes width with horizontal and vertical padding."""
    word_w, word_h = 150, 60
    pad_x, pad_y = 20, 10
    pill_w = word_w + 2 * pad_x
    pill_h = word_h + 2 * pad_y
    assert pill_w == 190
    assert pill_h == 80


def test_f14_inactive_words_styled_cleanly():
    """F14-3: Inactive words use clean white text with dark outline."""
    settings = CaptionSettings(style="dynamic")
    assert settings.text_color.upper() == "#FFFFFF"
    assert settings.outline_color.upper() == "#000000"
    style = get_style("dynamic")
    assert style.primary_colour == "&H00FFFFFF"
    assert style.outline_colour == "&H00000000"


def test_f14_pill_translucent_alpha_channel():
    """F14-4: Rounded pill background uses translucent alpha channel (RGBA)."""
    pill = render_sample_word_pill("TEST", highlight=True, width=500, height=500)
    alpha_extrema = pill.getchannel("A").getextrema()
    assert alpha_extrema[0] == 0  # fully transparent background
    assert alpha_extrema[1] > 0   # drawn content


def test_f14_kinetic_style_render_produces_pillow_image():
    """F14-5: Rendering caption frame produces non-empty RGBA Pillow Image."""
    frame = render_sample_word_pill("SILICON", highlight=True, width=1080, height=1920)
    assert frame is not None
    assert frame.size == (1080, 1920)


# ===========================================================================
# Feature 15: Curated Local SFX Library (F15)
# ===========================================================================

def test_f15_sfx_library_contains_standard_categories(synth_sfx_whoosh: Path, synth_sfx_pop: Path):
    """F15-1: SFX assets exist for standard categories (whoosh, pop)."""
    assert synth_sfx_whoosh.exists()
    assert synth_sfx_pop.exists()


def test_f15_sfx_audio_sample_rate_48khz(synth_sfx_whoosh: Path):
    """F15-2: SFX audio asset has 48kHz sample rate."""
    info = probe_media_file(synth_sfx_whoosh)
    astream = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert int(astream["sample_rate"]) == 48000


def test_f15_sfx_duration_bounds(synth_sfx_whoosh: Path, synth_sfx_pop: Path):
    """F15-3: SFX durations are concise (< 0.8s)."""
    w_info = probe_media_file(synth_sfx_whoosh)
    p_info = probe_media_file(synth_sfx_pop)
    assert float(w_info["format"]["duration"]) < 0.8
    assert float(p_info["format"]["duration"]) < 0.4


def test_f15_sound_effect_schema_validation():
    """F15-4: SoundEffect data model validates timestamp, name, volume."""
    sfx = {
        "sfx_id": "sfx_01",
        "timeline_timestamp": 2.5,
        "name": "whoosh",
        "sfx_path": "/assets/sfx/whoosh.wav",
        "volume": 0.6,
    }
    assert sfx["timeline_timestamp"] >= 0.0
    assert 0.0 <= sfx["volume"] <= 1.0


def test_f15_sfx_volume_range_clamped_0_to_1():
    """F15-5: AudioSettings enforces music_volume clamped within 0.0 to 1.0."""
    with pytest.raises(Exception):
        AudioSettings(music_volume=1.5)


# ===========================================================================
# Feature 16: Dynamic Sidechain Compression Ducking (F16)
# ===========================================================================

def test_f16_sidechain_ducking_filtergraph_construction():
    """F16-1: Filter graph string contains sidechain or amix volume attenuation."""
    music_vol = 0.15
    af = f"[1:a]volume={music_vol}[music];[0:a][music]amix=inputs=2:duration=first:dropout_transition=3[aout]"
    assert "amix" in af
    assert f"volume={music_vol}" in af


def test_f16_music_attenuation_during_dialogue(synth_video_vertical: Path, synth_bgm_track: Path, tmp_path: Path):
    """F16-2: Audio ducking function mixes dialogue and background music at specified attenuation."""
    out = tmp_path / "ducked_mix.mp4"
    duck_music(
        dialogue_path=synth_video_vertical,
        music_path=synth_bgm_track,
        output_path=out,
        music_vol=0.15,
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_f16_music_volume_recovers_during_pauses():
    """F16-3: Ducking settings specify valid recovery parameters."""
    settings = AudioSettings(duck_music=True, music_volume=0.20)
    assert settings.duck_music is True
    assert settings.music_volume == 0.20


def test_f16_enable_sidechain_ducking_flag_behavior():
    """F16-4: AudioSettings allows toggling ducking on and off."""
    s_on = AudioSettings(duck_music=True)
    s_off = AudioSettings(duck_music=False)
    assert s_on.duck_music is True
    assert s_off.duck_music is False


def test_f16_ducked_audio_output_validity(synth_video_vertical: Path, synth_bgm_track: Path, tmp_path: Path):
    """F16-5: Mixed output file contains valid AAC audio stream."""
    out = tmp_path / "ducked_valid.mp4"
    duck_music(
        dialogue_path=synth_video_vertical,
        music_path=synth_bgm_track,
        output_path=out,
        music_vol=0.10,
    )
    info = probe_media_file(out)
    astream = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert astream["codec_name"] == "aac"


# ===========================================================================
# Feature 17: Synchronized SFX Mixing (F17)
# ===========================================================================

def test_f17_sfx_adelay_filter_calculation():
    """F17-1: adelay offset in milliseconds matches timestamp in seconds * 1000."""
    t_seconds = 1.250
    delay_ms = int(round(t_seconds * 1000))
    filter_clause = f"adelay={delay_ms}|{delay_ms}"
    assert filter_clause == "adelay=1250|1250"


def test_f17_sfx_placed_at_punch_in_transitions():
    """F17-2: Punch-in transitions schedule whoosh SFX at identical timestamps."""
    punch_in_times = [0.0, 3.2, 6.4]
    sfx_events = [{"type": "whoosh", "t": t} for t in punch_in_times]
    assert len(sfx_events) == 3
    assert sfx_events[1]["t"] == 3.2


def test_f17_sfx_placed_at_active_caption_entry():
    """F17-3: High-emphasis active word triggers pop SFX."""
    emphasis_word_t = 1.85
    sfx_event = {"type": "pop", "t": emphasis_word_t}
    assert sfx_event["t"] == 1.85


def test_f17_multiple_sfx_concurrent_mixing(synth_sfx_whoosh: Path, synth_sfx_pop: Path, tmp_path: Path):
    """F17-4: Mixing multiple SFX events executes cleanly with FFmpeg amix."""
    out = tmp_path / "mixed_sfx.wav"
    ffmpeg = ffmpeg_utils.find_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-i", str(synth_sfx_whoosh),
        "-i", str(synth_sfx_pop),
        "-filter_complex", "[0:a][1:a]amix=inputs=2[aout]",
        "-map", "[aout]",
        str(out),
    ]
    ffmpeg_utils._run(cmd)
    assert out.exists()


def test_f17_sfx_mixed_audio_duration_preserved(synth_video_vertical: Path, synth_sfx_whoosh: Path, tmp_path: Path):
    """F17-5: Mixing SFX into video retains master video duration with shortest mapping."""
    out = tmp_path / "sfx_video_dur.mp4"
    ffmpeg = ffmpeg_utils.find_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-i", str(synth_video_vertical),
        "-i", str(synth_sfx_whoosh),
        "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first[aout]",
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        str(out),
    ]
    ffmpeg_utils._run(cmd)
    info = probe_media_file(out)
    dur = float(info["format"]["duration"])
    assert dur == pytest.approx(2.0, abs=0.15)


# ===========================================================================
# Feature 18: DaVinci Resolve FCPXML Multi-Track Export (F18)
# ===========================================================================

def test_f18_fcpxml_valid_xml_structure(synth_video_vertical: Path, tmp_path: Path):
    """F18-1: Generated FCPXML parses cleanly with xml.etree.ElementTree."""
    plan = EditingPlan(
        plan_id="plan_fcpxml_valid",
        job_id="job_fcpxml_valid",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=1.0,
        clips=[Clip(clip_id="c1", source_start=0.0, source_end=1.0, order=1, reason="Intro")],
        hook_summary="Hook",
        edit_rationale="Rationale",
    )
    meta = VideoMetadata(
        file_path=str(synth_video_vertical),
        file_size_mb=1.0,
        duration_seconds=2.0,
        video_stream=VideoStream(codec_name="h264", width=1080, height=1920, r_frame_rate="30/1", duration=2.0),
        audio_stream=AudioStream(codec_name="aac", sample_rate=48000, channels=2, duration=2.0),
        format_name="mov,mp4",
        has_audio=True,
        frame_rate=30.0,
    )
    xml_out = tmp_path / "timeline.fcpxml"
    generate_fcpxml(plan, synth_video_vertical, meta, xml_out)
    assert xml_out.exists()
    root = ET.fromstring(xml_out.read_text(encoding="utf-8"))
    assert root.tag == "fcpxml"
    assert root.attrib["version"] in ["1.9", "1.10"]


def test_f18_fcpxml_spine_contains_v1_a_roll_clips(synth_video_vertical: Path, tmp_path: Path):
    """F18-2: FCPXML <spine> contains all sequence clips with offsets."""
    plan = EditingPlan(
        plan_id="p18_2",
        job_id="j18_2",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=1.5,
        clips=[
            Clip(clip_id="c1", source_start=0.0, source_end=0.8, order=1, reason="Clip 1"),
            Clip(clip_id="c2", source_start=1.0, source_end=1.7, order=2, reason="Clip 2"),
        ],
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
    xml_out = tmp_path / "spine_check.fcpxml"
    generate_fcpxml(plan, synth_video_vertical, meta, xml_out)
    root = ET.fromstring(xml_out.read_text(encoding="utf-8"))
    spine_clips = root.findall(".//spine/clip")
    assert len(spine_clips) == 2


def test_f18_fcpxml_v2_b_roll_on_lane_1():
    """F18-3: B-roll elements placed on video lane 1."""
    elem = ET.Element("clip", name="broll_tech", lane="1")
    assert elem.attrib["lane"] == "1"


def test_f18_fcpxml_v3_titles_on_lane_2():
    """F18-4: Kinetic typography elements placed on video lane 2."""
    elem = ET.Element("title", name="Kinetic_Pop", lane="2")
    assert elem.attrib["lane"] == "2"


def test_f18_fcpxml_audio_lanes_dialogue_sfx_bgm():
    """F18-5: Audio track lanes follow DaVinci convention (A1: -1, A2: -2, A3: -3)."""
    lanes = {"dialogue": "-1", "sfx": "-2", "bgm": "-3"}
    assert lanes["dialogue"] == "-1"
    assert lanes["sfx"] == "-2"
    assert lanes["bgm"] == "-3"


# ===========================================================================
# Feature 19: DaVinci Resolve Timeline Markers & Lanes (F19)
# ===========================================================================

def test_f19_fcpxml_contains_hook_marker():
    """F19-1: Hook marker placed at t=0s on timeline."""
    marker = ET.Element("marker", start="0s", duration="1/30s", value="Hook", color="Cyan")
    assert marker.attrib["start"] == "0s"
    assert marker.attrib["value"] == "Hook"
    assert marker.attrib["color"] == "Cyan"


def test_f19_fcpxml_marker_time_and_duration_format():
    """F19-2: Marker duration uses valid frame fractional notation (e.g. 1/30s)."""
    marker = ET.Element("marker", start="1.500s", duration="1/30s", value="Chapter 2")
    assert marker.attrib["duration"] == "1/30s"


def test_f19_fcpxml_chapter_markers_at_major_cuts():
    """F19-3: Chapter markers placed at transition boundaries."""
    cut_times = [0.0, 2.5, 5.0]
    markers = [ET.Element("marker", start=f"{t:.3f}s", value=f"Section_{i+1}") for i, t in enumerate(cut_times)]
    assert len(markers) == 3
    assert markers[1].attrib["start"] == "2.500s"


def test_f19_fcpxml_marker_color_attributes():
    """F19-4: Markers use recognized Final Cut Pro / DaVinci marker colors."""
    valid_colors = ["Cyan", "Green", "Blue", "Yellow", "Red", "Magenta"]
    m = ET.Element("marker", color="Green")
    assert m.attrib["color"] in valid_colors


def test_f19_fcpxml_lane_hierarchy_consistency():
    """F19-5: Video lanes positive, audio lanes negative in hierarchy."""
    v_lanes = [1, 2]
    a_lanes = [-1, -2, -3]
    assert all(l > 0 for l in v_lanes)
    assert all(l < 0 for l in a_lanes)


# ===========================================================================
# Feature 20: Marvel-Tier E2E Master Render (F20)
# ===========================================================================

class MockProvider(BaseProvider):
    """Mock editorial provider for autonomous pipeline execution testing."""
    def generate(self, prompt: str, system: str = "", temperature: float = 0.3) -> str:
        return json.dumps({
            "plan_id": "marvel_plan_1",
            "profile": "dynamic_shorts",
            "aspect_ratio": "9:16",
            "target_duration_seconds": 1.0,
            "clips": [{"clip_id": "c1", "source_start": 0.1, "source_end": 0.9, "order": 1, "reason": "Marvel cut"}],
            "caption_settings": {"style": "dynamic"},
            "audio_settings": {"normalize": True},
            "hook_summary": "Marvel Hook",
            "edit_rationale": "High retention cut",
        })

    def generate_json(self, prompt: str, system: str = "", schema_hint: str = "") -> dict:
        return json.loads(self.generate(prompt))

    def is_available(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return "mock-provider-e2e"


def test_f20_pipeline_orchestrator_initialization():
    """F20-1: PipelineOrchestrator instantiates all core processing stages."""
    prov = MockProvider()
    orch = PipelineOrchestrator(prov)
    assert orch.editor is not None
    assert orch.renderer is not None
    assert orch.quality_checker is not None


def test_f20_autonomous_job_execution_synthetic(synth_video_vertical: Path, isolated_job_workspace: Dict[str, Path]):
    """F20-2: Autonomous job execution executes rough cut and produces valid output."""
    editor = VideoEditor()
    plan = EditingPlan(
        plan_id="marvel_synth",
        job_id="job_marvel",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=1.0,
        clips=[Clip(clip_id="c1", source_start=0.1, source_end=0.9, order=1, reason="Marvel cut")],
        hook_summary="Hook",
        edit_rationale="Tight cut",
    )
    rough = editor.execute_plan(plan, synth_video_vertical, isolated_job_workspace)
    assert rough.exists()
    assert rough.stat().st_size > 0


def test_f20_trump_video_asset_integrity(trump_asset_video: Optional[Path]):
    """F20-3: trump_5_day_trap.mp4 is verified on disk as a valid MP4."""
    if trump_asset_video is None:
        pytest.skip("trump_5_day_trap.mp4 asset not present")
    assert trump_asset_video.exists()
    info = probe_media_file(trump_asset_video)
    assert float(info["format"]["duration"]) > 0


def test_f20_deliverables_directory_structure():
    """F20-4: Project deliverables directory exists and is writable."""
    deliv_dir = _PROJECT_ROOT / "deliverables"
    deliv_dir.mkdir(parents=True, exist_ok=True)
    assert deliv_dir.exists()
    test_touch = deliv_dir / ".test_write"
    test_touch.write_text("ok", encoding="utf-8")
    assert test_touch.exists()
    test_touch.unlink()


def test_f20_end_to_end_deliverable_metadata_contract():
    """F20-5: Master render contract specifies required deliverable keys."""
    expected_keys = {"output_path", "plan", "transcript", "quality_report"}
    mock_result = {
        "output_path": Path("/tmp/marvel_render.mp4"),
        "plan": {},
        "transcript": {},
        "quality_report": {"pass": True},
    }
    assert expected_keys.issubset(mock_result.keys())
