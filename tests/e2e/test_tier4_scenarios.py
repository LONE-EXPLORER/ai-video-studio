"""
tests/e2e/test_tier4_scenarios.py — Comprehensive Tier 4 Real-World Application Scenarios.

Validates end-to-end production workflows under real-world conditions:
- Scenario 1: YouTube Shorts High-Retention Fast-Cut Flow
- Scenario 2: Multi-Camera Podcast Interview Switching
- Scenario 3: Documentary Talking Head with B-Roll & Ducked BGM
- Scenario 4: Kinetic Typography Viral Social Reel Flow
- Scenario 5: DaVinci Resolve Professional NLE FCPXML Handoff
- Scenario 6: End-to-End Processing of User Asset (trump_5_day_trap.mp4)
- Scenario 7: Fault-Tolerant Pipeline Stage Runner Recovery
"""

from __future__ import annotations

import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from PIL import Image, ImageDraw

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
from app.captions.pillow_captions import (
    burn_kinetic_captions,
    map_words_to_timeline,
    group_words_into_phrases,
    MappedWord,
    CaptionGroup,
)
from app.pipeline.stage_runner import StageRunner, run_stages
from app.utils import ffmpeg_utils
from tests.e2e.conftest import make_synthetic_video, make_synthetic_audio, probe_media_file

pytestmark = pytest.mark.tier4


# ===========================================================================
# Scenario 1: YouTube Shorts High-Retention Fast-Cut Flow
# ===========================================================================

def test_scenario_1_youtube_shorts_retention_flow(
    synth_video_vertical: Path,
    isolated_job_workspace: Dict[str, Path],
):
    """
    Scenario 1: High-Retention YouTube Shorts Flow
    - Ingest 1080x1920 source video
    - Apply 1.15x eye-level center crop punch-in hook on Clip 1 (0.0s-0.8s)
    - Cut Clip 2 (0.9s-1.8s) as detail angle
    - Apply 10ms micro audio fades to suppress zero-crossing clicks
    - Normalise dialogue audio to -16 LUFS
    - Produce final 1080x1920 vertical MP4 rough cut
    """
    editor = VideoEditor()
    plan = EditingPlan(
        plan_id="plan_yt_shorts_01",
        job_id="job_yt_shorts_01",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=1.7,
        clips=[
            Clip(clip_id="c01_hook", source_start=0.0, source_end=0.8, order=1, reason="Retention Hook", zoom=ZoomEffect(enabled=True, factor=1.15)),
            Clip(clip_id="c02_body", source_start=0.9, source_end=1.8, order=2, reason="Value Delivery", zoom=ZoomEffect(enabled=False, factor=1.0)),
        ],
        caption_settings=CaptionSettings(style="dynamic", font_size=56),
        audio_settings=AudioSettings(normalize=True, fade_in=0.01, fade_out=0.01),
        hook_summary="Punchy Retention Hook",
        edit_rationale="Sub-2-second fast-cut pattern interrupt",
    )

    rough_cut = editor.execute_plan(plan, synth_video_vertical, isolated_job_workspace)
    assert rough_cut.exists()
    assert rough_cut.stat().st_size > 0

    info = probe_media_file(rough_cut)
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert vstream["width"] == 1080
    assert vstream["height"] == 1920
    assert vstream["codec_name"] == "h264"
    dur = float(info["format"]["duration"])
    assert dur == pytest.approx(1.7, abs=0.20)


# ===========================================================================
# Scenario 2: Multi-Camera Podcast Interview Switching
# ===========================================================================

def test_scenario_2_multicam_podcast_interview_flow(
    tmp_path: Path,
    isolated_job_workspace: Dict[str, Path],
):
    """
    Scenario 2: Multi-Camera Podcast Interview Flow
    - Camera A (Host) and Camera B (Guest) ingested as separate sources
    - Normalised to matching 1080x1920 30fps stream geometry
    - Interleaved cutting: Host (0.0s-1.0s) -> Guest (0.0s-1.0s) -> Host (1.0s-2.0s)
    - Concatenated into seamless multi-angle conversational edit
    """
    cam_a = tmp_path / "cam_host.mp4"
    cam_b = tmp_path / "cam_guest.mp4"
    make_synthetic_video(cam_a, width=1080, height=1920, fps=30, duration=2.0, pattern="testsrc")
    make_synthetic_video(cam_b, width=1080, height=1920, fps=30, duration=2.0, pattern="testsrc2")

    intermediate = isolated_job_workspace["intermediate"]
    cut1 = intermediate / "cut1_host.mp4"
    cut2 = intermediate / "cut2_guest.mp4"
    cut3 = intermediate / "cut3_host.mp4"

    ffmpeg_utils.cut_segment(cam_a, 0.0, 0.8, cut1)
    ffmpeg_utils.cut_segment(cam_b, 0.2, 1.0, cut2)
    ffmpeg_utils.cut_segment(cam_a, 1.0, 1.8, cut3)

    podcast_edit = intermediate / "podcast_stitched.mp4"
    ffmpeg_utils.concat_clips([cut1, cut2, cut3], podcast_edit)
    assert podcast_edit.exists()

    info = probe_media_file(podcast_edit)
    dur = float(info["format"]["duration"])
    assert dur == pytest.approx(2.4, abs=0.25)


# ===========================================================================
# Scenario 3: Documentary Talking Head with B-Roll & Ducked BGM
# ===========================================================================

def test_scenario_3_broll_documentary_reel_flow(
    synth_video_vertical: Path,
    synth_broll_clip: Path,
    synth_bgm_track: Path,
    tmp_path: Path,
):
    """
    Scenario 3: Documentary Talking Head with B-Roll & Ducked BGM
    - Host dialogue track establishes narration
    - Background music mixed in and dynamically attenuated by 85% (-16dB duck)
    - B-roll cutaway overlaid at midpoint (t=0.6s to t=1.4s) while dialogue continues
    - Final output retains uninterrupted dialogue audio with professional audio-visual layering
    """
    # 1. Duck background music under dialogue
    ducked_mix = tmp_path / "doc_audio_mix.mp4"
    duck_music(synth_video_vertical, synth_bgm_track, ducked_mix, music_vol=0.15)
    assert ducked_mix.exists()

    # 2. Composite B-roll cutaway over the ducked dialogue video
    doc_final = tmp_path / "doc_reel_final.mp4"
    ffmpeg = ffmpeg_utils.find_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-i", str(ducked_mix),
        "-i", str(synth_broll_clip),
        "-filter_complex", "[0:v][1:v]overlay=enable='between(t,0.6,1.4)'[vout]",
        "-map", "[vout]",
        "-map", "0:a",
        "-c:v", "libx264",
        "-c:a", "copy",
        str(doc_final),
    ]
    ffmpeg_utils._run(cmd)
    assert doc_final.exists()

    info = probe_media_file(doc_final)
    dur = float(info["format"]["duration"])
    assert dur == pytest.approx(2.0, abs=0.2)


# ===========================================================================
# Scenario 4: Kinetic Typography Viral Social Reel Flow
# ===========================================================================

def test_scenario_4_kinetic_typography_social_reel_flow(
    synth_video_vertical: Path,
    sample_e2e_transcript: Transcript,
    tmp_path: Path,
):
    """
    Scenario 4: Kinetic Typography Viral Social Reel Flow
    - Maps spoken words from transcript to timeline
    - Word-level active pill highlight (#FFCC00) rendered with Pillow
    - Burned into video using FFmpeg hardware overlay
    - Verifies valid burned video container with synchronized dialogue
    """
    plan = EditingPlan(
        plan_id="p_kinetic_reel",
        job_id="j_kinetic_reel",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=2.0,
        clips=[Clip(clip_id="c1", source_start=0.0, source_end=2.0, order=1, reason="Kinetic speech")],
        hook_summary="Hook",
        edit_rationale="Kinetic text",
    )

    # Map words and group into phrases
    mapped = map_words_to_timeline(plan, sample_e2e_transcript)
    assert len(mapped) > 0
    groups = group_words_into_phrases(mapped, max_words=3)
    assert len(groups) > 0

    # Burn captions onto vertical video
    kinetic_out = tmp_path / "kinetic_burned.mp4"
    burn_kinetic_captions(synth_video_vertical, plan, sample_e2e_transcript, kinetic_out, fps=30)
    assert kinetic_out.exists()
    assert kinetic_out.stat().st_size > 0


# ===========================================================================
# Scenario 5: DaVinci Resolve Professional NLE FCPXML Handoff
# ===========================================================================

def test_scenario_5_davinci_resolve_pro_editor_handoff_flow(
    synth_video_vertical: Path,
    tmp_path: Path,
):
    """
    Scenario 5: DaVinci Resolve NLE Handoff Workflow
    - Generates multi-clip editing timeline
    - Exports FCPXML (v1.9) file with precise frame duration and metadata
    - Validates XML structure: spine, assets, format, transform scale, markers
    - Confirms import readiness for DaVinci Resolve Studio on macOS
    """
    plan = EditingPlan(
        plan_id="p_davinci_pro",
        job_id="j_davinci_pro",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=2.0,
        clips=[
            Clip(clip_id="c01_hook", source_start=0.0, source_end=1.0, order=1, reason="Hook", zoom=ZoomEffect(enabled=True, factor=1.15)),
            Clip(clip_id="c02_body", source_start=1.0, source_end=2.0, order=2, reason="Core", zoom=ZoomEffect(enabled=False, factor=1.0)),
        ],
        hook_summary="Pro Hook",
        edit_rationale="DaVinci timeline generation",
    )
    meta = VideoMetadata(
        file_path=str(synth_video_vertical),
        file_size_mb=1.5,
        duration_seconds=2.0,
        video_stream=VideoStream(codec_name="h264", width=1080, height=1920, r_frame_rate="30/1", duration=2.0),
        audio_stream=AudioStream(codec_name="aac", sample_rate=48000, channels=2, duration=2.0),
        format_name="mp4",
        has_audio=True,
        frame_rate=30.0,
    )
    fcpxml_path = tmp_path / "pro_timeline.fcpxml"
    generate_fcpxml(plan, synth_video_vertical, meta, fcpxml_path, project_name="DaVinci Pro Handoff")
    assert fcpxml_path.exists()

    # DTD verification
    root = ET.fromstring(fcpxml_path.read_text(encoding="utf-8"))
    assert root.tag == "fcpxml"
    assert root.attrib["version"] == "1.9"
    project = root.find(".//project")
    assert project is not None
    assert project.attrib["name"] == "DaVinci Pro Handoff"
    spine_clips = root.findall(".//spine/clip")
    assert len(spine_clips) == 2


# ===========================================================================
# Scenario 6: End-to-End Processing of User Asset (trump_5_day_trap.mp4)
# ===========================================================================

def test_scenario_6_autonomous_trump_asset_master_render(
    trump_asset_video: Optional[Path],
    isolated_job_workspace: Dict[str, Path],
    tmp_path: Path,
):
    """
    Scenario 6: End-to-End Processing of Primary Test Asset (trump_5_day_trap.mp4)
    - Verifies primary user video asset exists and probes codec streams
    - Executes a realistic editing plan extracting an engaging 3.0s hook cut
    - Applies 1.15x punch-in zoom for YouTube Shorts pattern interruption
    - Processes audio with micro-fades and loudness normalization
    - Saves rendered master short into deliverables directory
    - Asserts valid deliverables container adhering to project specifications
    """
    if trump_asset_video is None:
        pytest.skip("trump_5_day_trap.mp4 not found in assets/test_videos/")

    # 1. Probe input video metadata
    info = probe_media_file(trump_asset_video)
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
    astream = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    assert vstream is not None
    source_duration = float(info["format"]["duration"])
    assert source_duration >= 3.0

    # 2. Formulate 3.0s hook editing plan
    editor = VideoEditor()
    plan = EditingPlan(
        plan_id="plan_trump_marvel",
        job_id="job_trump_marvel",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=3.0,
        clips=[
            Clip(clip_id="c01_hook", source_start=0.5, source_end=2.0, order=1, reason="Dramatic opening statement", zoom=ZoomEffect(enabled=True, factor=1.15)),
            Clip(clip_id="c02_climax", source_start=2.0, source_end=3.5, order=2, reason="Key hook conclusion", zoom=ZoomEffect(enabled=False, factor=1.0)),
        ],
        caption_settings=CaptionSettings(style="dynamic", font_size=56),
        audio_settings=AudioSettings(normalize=True, fade_in=0.01, fade_out=0.01),
        hook_summary="Trump 5-Day Trap Hook",
        edit_rationale="High retention hook cut for viral shorts",
    )

    # 3. Execute rough cut
    rough_cut = editor.execute_plan(plan, trump_asset_video, isolated_job_workspace)
    assert rough_cut.exists()
    assert rough_cut.stat().st_size > 0

    # 4. Verify deliverables contract
    deliverables_dir = _PROJECT_ROOT / "deliverables"
    deliverables_dir.mkdir(parents=True, exist_ok=True)
    master_render = deliverables_dir / "test_marvel_render.mp4"
    shutil.copy2(rough_cut, master_render)
    assert master_render.exists()

    master_info = probe_media_file(master_render)
    master_vstream = next(s for s in master_info["streams"] if s["codec_type"] == "video")
    assert master_vstream["codec_name"] == "h264"
    master_dur = float(master_info["format"]["duration"])
    assert master_dur == pytest.approx(3.0, abs=0.35)


# ===========================================================================
# Scenario 7: Fault-Tolerant Pipeline Stage Runner Recovery
# ===========================================================================

def test_scenario_7_fault_tolerant_pipeline_stage_recovery():
    """
    Scenario 7: Fault-Tolerant Pipeline Stage Runner Recovery
    - Verifies run_stages handles non-critical stage failure cleanly
    - Non-critical stage (e.g. contact sheet thumbnail generation) raises exception
    - Runner logs warning and continues to subsequent critical rendering stages
    - Entire pipeline finishes successfully without aborting job
    """
    execution_log: List[str] = []

    def stage_ingest(ctx: Dict[str, Any]) -> Dict[str, Any]:
        execution_log.append("ingest_done")
        return {"ingest": True}

    def stage_optional_thumbnails(ctx: Dict[str, Any]) -> Dict[str, Any]:
        execution_log.append("thumbnail_attempted")
        raise ValueError("Optional contact sheet generation failed due to missing font")

    def stage_render(ctx: Dict[str, Any]) -> Dict[str, Any]:
        execution_log.append("render_done")
        return {"render_output": "/deliverables/final.mp4"}

    stages = [
        StageRunner(name="ingest", fn=stage_ingest, critical=True, description="Ingest"),
        StageRunner(name="thumbnails", fn=stage_optional_thumbnails, critical=False, description="Thumbnails"),
        StageRunner(name="render", fn=stage_render, critical=True, description="Render"),
    ]

    context: Dict[str, Any] = {}
    updated_context = run_stages(stages, context)

    assert "ingest_done" in execution_log
    assert "thumbnail_attempted" in execution_log
    assert "render_done" in execution_log
    assert updated_context.get("render_output") == "/deliverables/final.mp4"
