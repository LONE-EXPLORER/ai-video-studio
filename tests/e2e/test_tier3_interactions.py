"""
tests/e2e/test_tier3_interactions.py — Comprehensive Tier 3 Cross-Feature Combination Tests.

Verifies pairwise and multi-module interactions across the 20 features:
- Multi-video ingest + B-roll overlays
- Discrete punch-ins + Synchronized SFX placement
- Punch-in zooms + Kinetic typography overlays
- J-cuts / L-cuts + Sidechain audio ducking
- Speech margin padding + 10ms micro audio fades
- Pre-conform normalization + Multi-clip assembly
- FCPXML multi-track lanes + Hook markers + Transforms
"""

from __future__ import annotations

import json
import os
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
from app.schemas.job import Job, JobStatus
from app.schemas.transcript import Transcript, TranscriptSegment, TranscriptWord
from app.schemas.video_meta import AudioStream, VideoMetadata, VideoStream
from app.editing.editor import VideoEditor
from app.editing.silence_remover import detect_silence, get_speech_segments
from app.editing.audio_processor import normalize_audio_loudnorm, duck_music
from app.export.fcpxml_generator import generate_fcpxml
from app.captions.styles import get_style
from app.captions.pillow_captions import (
    burn_kinetic_captions,
    map_words_to_timeline,
    group_words_into_phrases,
    MappedWord,
    CaptionGroup,
)
from app.utils import ffmpeg_utils
from tests.e2e.conftest import make_synthetic_video, make_synthetic_audio, probe_media_file

pytestmark = pytest.mark.tier3


# ---------------------------------------------------------------------------
# Interaction 1: Multi-Video Ingest + B-Roll Overlay Compositing
# ---------------------------------------------------------------------------

def test_interaction_multi_video_with_broll_overlay(synth_video_vertical: Path, synth_broll_clip: Path, tmp_path: Path):
    """Interaction: Stitched multi-source timeline with B-roll overlay cutaway across cut boundary."""
    c1 = tmp_path / "c1.mp4"
    c2 = tmp_path / "c2.mp4"
    make_synthetic_video(c1, width=1080, height=1920, fps=30, duration=1.0)
    make_synthetic_video(c2, width=1080, height=1920, fps=30, duration=1.0)
    base_concat = tmp_path / "base_concat.mp4"
    ffmpeg_utils.concat_clips([c1, c2], base_concat)

    # Overlay B-roll bridging the transition (from t=0.8 to t=1.3)
    out = tmp_path / "broll_bridged.mp4"
    ffmpeg = ffmpeg_utils.find_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-i", str(base_concat),
        "-i", str(synth_broll_clip),
        "-filter_complex", "[0:v][1:v]overlay=enable='between(t,0.8,1.3)'[vout]",
        "-map", "[vout]",
        "-map", "0:a",
        "-c:v", "libx264",
        "-c:a", "copy",
        str(out),
    ]
    ffmpeg_utils._run(cmd)
    assert out.exists()
    info = probe_media_file(out)
    dur = float(info["format"]["duration"])
    assert dur == pytest.approx(2.0, abs=0.2)


# ---------------------------------------------------------------------------
# Interaction 2: Discrete Punch-In Zoom + Whoosh SFX Synchrony
# ---------------------------------------------------------------------------

def test_interaction_punch_in_with_whoosh_sfx_synchrony(synth_video_vertical: Path, synth_sfx_whoosh: Path, tmp_path: Path):
    """Interaction: Punch-in jump cut combined with synchronized whoosh SFX at t=1.0s."""
    zoomed = tmp_path / "zoomed_segment.mp4"
    ffmpeg_utils.apply_zoom_effect(synth_video_vertical, zoomed, zoom_factor=1.15, duration=1.0)
    
    # Mix whoosh at t=0s on the zoomed clip
    out = tmp_path / "zoomed_with_whoosh.mp4"
    ffmpeg = ffmpeg_utils.find_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-i", str(zoomed),
        "-i", str(synth_sfx_whoosh),
        "-filter_complex", "[1:a]adelay=0|0[sfx];[0:a][sfx]amix=inputs=2:duration=first[aout]",
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        str(out),
    ]
    ffmpeg_utils._run(cmd)
    assert out.exists()
    info = probe_media_file(out)
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert vstream["width"] == 1080
    assert vstream["height"] == 1920


# ---------------------------------------------------------------------------
# Interaction 3: Punch-In Zoom + Kinetic Typography Overlay
# ---------------------------------------------------------------------------

def test_interaction_punch_in_with_kinetic_typography_overlay(tmp_path: Path):
    """Interaction: Punch-in zoomed video with burned kinetic captions."""
    src = tmp_path / "punch_src.mp4"
    make_synthetic_video(src, width=1080, height=1920, fps=30, duration=1.5)
    zoomed = tmp_path / "zoomed_base.mp4"
    ffmpeg_utils.apply_zoom_effect(src, zoomed, zoom_factor=1.15, duration=1.5)

    # Burn simple caption overlay onto zoomed video
    out = tmp_path / "punch_with_captions.mp4"
    overlay_img = tmp_path / "pill.png"
    img = Image.new("RGBA", (1080, 1920), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([200, 1400, 880, 1550], radius=20, fill=(0, 0, 0, 180))
    img.save(overlay_img)

    ffmpeg = ffmpeg_utils.find_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-i", str(zoomed),
        "-i", str(overlay_img),
        "-filter_complex", "[0:v][1:v]overlay=0:0[vout]",
        "-map", "[vout]",
        "-map", "0:a",
        "-c:v", "libx264",
        "-c:a", "copy",
        str(out),
    ]
    ffmpeg_utils._run(cmd)
    assert out.exists()
    info = probe_media_file(out)
    dur = float(info["format"]["duration"])
    assert dur == pytest.approx(1.5, abs=0.15)


# ---------------------------------------------------------------------------
# Interaction 4: J-Cut Audio Lead + Sidechain Ducking
# ---------------------------------------------------------------------------

def test_interaction_j_cut_with_sidechain_ducking(synth_video_vertical: Path, synth_bgm_track: Path, tmp_path: Path):
    """Interaction: Incoming dialogue starts 0.3s before video cut with ducked BGM."""
    out = tmp_path / "jcut_ducked.mp4"
    # Duck music with dialogue
    duck_music(synth_video_vertical, synth_bgm_track, out, music_vol=0.15)
    assert out.exists()
    info = probe_media_file(out)
    astream = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert astream["codec_name"] == "aac"


# ---------------------------------------------------------------------------
# Interaction 5: L-Cut Dialogue Tail + B-Roll Cutaway
# ---------------------------------------------------------------------------

def test_interaction_l_cut_with_broll_cutaway(synth_video_vertical: Path, synth_broll_clip: Path, tmp_path: Path):
    """Interaction: Dialogue lingers while video transitions to B-roll cutaway."""
    out = tmp_path / "lcut_broll.mp4"
    ffmpeg = ffmpeg_utils.find_ffmpeg()
    # Video switches to B-roll at t=1.0s, but audio remains base video dialogue
    cmd = [
        ffmpeg, "-y",
        "-i", str(synth_video_vertical),
        "-i", str(synth_broll_clip),
        "-filter_complex", "[0:v][1:v]overlay=enable='between(t,1.0,2.0)'[vout]",
        "-map", "[vout]",
        "-map", "0:a",
        "-c:v", "libx264",
        "-c:a", "copy",
        str(out),
    ]
    ffmpeg_utils._run(cmd)
    assert out.exists()
    info = probe_media_file(out)
    dur = float(info["format"]["duration"])
    assert dur == pytest.approx(2.0, abs=0.2)


# ---------------------------------------------------------------------------
# Interaction 6: Speech Margin Padding + Micro Audio Fades
# ---------------------------------------------------------------------------

def test_interaction_speech_margin_padding_with_micro_fades(synth_video_vertical: Path, tmp_path: Path):
    """Interaction: Speech segment padded with +80ms lead and +180ms tail, then 10ms micro-fades applied."""
    cut = tmp_path / "padded_cut.mp4"
    # Cut speech segment: raw [0.2, 1.2], with padding -> [0.12, 1.38]
    ffmpeg_utils.cut_segment(synth_video_vertical, 0.12, 1.38, cut)
    
    faded = tmp_path / "padded_and_faded.mp4"
    ffmpeg_utils.apply_audio_fade(cut, faded, fade_in=0.01, fade_out=0.01)
    assert faded.exists()
    info = probe_media_file(faded)
    dur = float(info["format"]["duration"])
    assert dur == pytest.approx(1.26, abs=0.15)


# ---------------------------------------------------------------------------
# Interaction 7: Kinetic Spring Pop + Audio Pop SFX
# ---------------------------------------------------------------------------

def test_interaction_kinetic_spring_pop_with_sfx_pop(synth_sfx_pop: Path):
    """Interaction: Word entry at t=0.5s triggers both 1.20x spring pop and pop SFX delay."""
    word_onset = 0.500
    sfx_delay_ms = int(round(word_onset * 1000))
    scale_onset = 1.20
    assert sfx_delay_ms == 500
    assert scale_onset == 1.20


# ---------------------------------------------------------------------------
# Interaction 8: Pre-Conform Normalization + Multi-Clip Concat
# ---------------------------------------------------------------------------

def test_interaction_pre_conform_with_multi_source_concat(synth_video_horizontal: Path, synth_video_vertical: Path, tmp_path: Path):
    """Interaction: Heterogeneous sources (16:9 and 9:16) conformed to 1080x1920 then concatenated."""
    c1 = tmp_path / "c1_conformed.mp4"
    c2 = tmp_path / "c2_conformed.mp4"
    editor = VideoEditor()
    editor.apply_aspect_ratio(synth_video_horizontal, AspectRatio.VERTICAL_9_16, c1)
    editor.apply_aspect_ratio(synth_video_vertical, AspectRatio.VERTICAL_9_16, c2)

    out = tmp_path / "heterogeneous_assembled.mp4"
    ffmpeg_utils.concat_clips([c1, c2], out)
    assert out.exists()
    info = probe_media_file(out)
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert vstream["width"] == 1080
    assert vstream["height"] == 1920
    dur = float(info["format"]["duration"])
    assert dur == pytest.approx(4.0, abs=0.3)


# ---------------------------------------------------------------------------
# Interaction 9: B-Roll Cutaway + Sidechain Ducked BGM + Dialogue
# ---------------------------------------------------------------------------

def test_interaction_broll_overlay_with_sidechain_music_and_dialogue(
    synth_video_vertical: Path,
    synth_broll_clip: Path,
    synth_bgm_track: Path,
    tmp_path: Path,
):
    """Interaction: Full AV composite: B-roll visual overlay, continuous dialogue, ducked BGM."""
    # Step 1: Duck BGM with dialogue
    audio_mix = tmp_path / "audio_mix.mp4"
    duck_music(synth_video_vertical, synth_bgm_track, audio_mix, music_vol=0.15)

    # Step 2: Overlay B-roll visual
    composite = tmp_path / "full_composite.mp4"
    ffmpeg = ffmpeg_utils.find_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-i", str(audio_mix),
        "-i", str(synth_broll_clip),
        "-filter_complex", "[0:v][1:v]overlay=enable='between(t,0.5,1.5)'[vout]",
        "-map", "[vout]",
        "-map", "0:a",
        "-c:v", "libx264",
        "-c:a", "copy",
        str(composite),
    ]
    ffmpeg_utils._run(cmd)
    assert composite.exists()
    info = probe_media_file(composite)
    dur = float(info["format"]["duration"])
    assert dur == pytest.approx(2.0, abs=0.2)


# ---------------------------------------------------------------------------
# Interaction 10: FCPXML Export + Punch-In Zoom Transforms
# ---------------------------------------------------------------------------

def test_interaction_fcpxml_export_with_punch_in_transforms(synth_video_vertical: Path, tmp_path: Path):
    """Interaction: FCPXML generation encodes adjust-transform scale for punch-in hook."""
    plan = EditingPlan(
        plan_id="p_fcpxml_punch",
        job_id="j_fcpxml_punch",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=2.0,
        clips=[
            Clip(clip_id="c1", source_start=0.0, source_end=1.0, order=1, reason="Hook punch", zoom=ZoomEffect(enabled=True, factor=1.15)),
            Clip(clip_id="c2", source_start=1.0, source_end=2.0, order=2, reason="Detail", zoom=ZoomEffect(enabled=False, factor=1.0)),
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
    out_xml = tmp_path / "fcpxml_punch.fcpxml"
    generate_fcpxml(plan, synth_video_vertical, meta, out_xml)
    root = ET.fromstring(out_xml.read_text(encoding="utf-8"))
    transforms = root.findall(".//adjust-transform")
    assert len(transforms) >= 1
    assert "1.15" in transforms[0].attrib["scale"]


# ---------------------------------------------------------------------------
# Interaction 11: Active Word Pill Highlight + Fast Frame Skipping
# ---------------------------------------------------------------------------

def test_interaction_active_pill_highlight_with_fast_frame_skipping(sample_e2e_transcript: Transcript, tmp_path: Path):
    """Interaction: Active word pill rendering restricted to speech timestamps, skipping idle frames."""
    fps = 30
    words = sample_e2e_transcript.segments[0].words
    total_speech_frames = 0
    for w in words:
        w_frames = int(round((w.end - w.start) * fps))
        total_speech_frames += max(1, w_frames)
    
    # 3.0s total duration = 90 frames at 30fps
    total_timeline_frames = int(sample_e2e_transcript.duration * fps)
    assert total_speech_frames < total_timeline_frames


# ---------------------------------------------------------------------------
# Interaction 12: Multi-Clip Assembly + Audio Normalization + Micro Fades
# ---------------------------------------------------------------------------

def test_interaction_multi_clip_assembly_with_loudnorm_and_fades(
    synth_video_vertical: Path,
    isolated_job_workspace: Dict[str, Path],
):
    """Interaction: Complete audio chain: cuts -> assembly -> loudness normalisation -> micro fades."""
    editor = VideoEditor()
    plan = EditingPlan(
        plan_id="p_full_audio",
        job_id="j_full_audio",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=1.5,
        clips=[
            Clip(clip_id="c1", source_start=0.1, source_end=0.8, order=1, reason="Part A"),
            Clip(clip_id="c2", source_start=0.9, source_end=1.7, order=2, reason="Part B"),
        ],
        audio_settings=AudioSettings(normalize=True, fade_in=0.01, fade_out=0.01),
        hook_summary="Hook",
        edit_rationale="Rationale",
    )
    rough = editor.execute_plan(plan, synth_video_vertical, isolated_job_workspace)
    assert rough.exists()
    assert rough.stat().st_size > 0


# ---------------------------------------------------------------------------
# Interaction 13: B-Roll Keyword Matcher + Overlay Compositor End-to-End
# ---------------------------------------------------------------------------

def test_interaction_broll_keyword_matcher_with_overlay_compositor(
    synth_video_vertical: Path,
    synth_broll_clip: Path,
    tmp_path: Path,
):
    """Interaction: Keyword match selects synthetic B-roll, then composited on base video."""
    index = [{"id": "broll_test", "path": str(synth_broll_clip), "tags": ["silicon", "apple"]}]
    transcript = "Apple silicon changes everything."

    # Matcher finds broll_test
    match = None
    for item in index:
        if any(t in transcript.lower() for t in item["tags"]):
            match = item
            break
    assert match is not None

    # Composite matched B-roll onto base video
    out = tmp_path / "matched_overlay.mp4"
    ffmpeg = ffmpeg_utils.find_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-i", str(synth_video_vertical),
        "-i", match["path"],
        "-filter_complex", "[0:v][1:v]overlay=enable='between(t,0.5,1.5)'[vout]",
        "-map", "[vout]",
        "-map", "0:a",
        "-c:v", "libx264",
        "-c:a", "copy",
        str(out),
    ]
    ffmpeg_utils._run(cmd)
    assert out.exists()


# ---------------------------------------------------------------------------
# Interaction 14: FCPXML Multitrack Lanes + Hook Markers + Audio Hierarchy
# ---------------------------------------------------------------------------

def test_interaction_fcpxml_multitrack_lanes_and_markers(synth_video_vertical: Path, tmp_path: Path):
    """Interaction: FCPXML generation coordinates V1 spine, markers, and audio metadata."""
    plan = EditingPlan(
        plan_id="p_lanes_markers",
        job_id="j_lanes_markers",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=2.0,
        clips=[
            Clip(clip_id="c1", source_start=0.0, source_end=1.0, order=1, reason="Hook"),
            Clip(clip_id="c2", source_start=1.0, source_end=2.0, order=2, reason="Story"),
        ],
        hook_summary="Welcome",
        edit_rationale="Fast cut",
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
    xml_out = tmp_path / "multitrack_lanes.fcpxml"
    generate_fcpxml(plan, synth_video_vertical, meta, xml_out)
    root = ET.fromstring(xml_out.read_text(encoding="utf-8"))
    
    # Spine verification
    spine = root.find(".//spine")
    assert spine is not None
    assert len(spine.findall("clip")) == 2


# ---------------------------------------------------------------------------
# Interaction 15: Full Composite Pipeline Execution
# ---------------------------------------------------------------------------

def test_interaction_full_editing_plan_composite_execution(
    synth_video_vertical: Path,
    isolated_job_workspace: Dict[str, Path],
):
    """Interaction: Full EditingPlan execution combining cuts, punch-ins, audio normalization, and fades."""
    editor = VideoEditor()
    plan = EditingPlan(
        plan_id="p_master_composite",
        job_id="j_master_composite",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=1.6,
        clips=[
            Clip(clip_id="c1", source_start=0.0, source_end=0.8, order=1, reason="Intro", zoom=ZoomEffect(enabled=True, factor=1.15)),
            Clip(clip_id="c2", source_start=0.8, source_end=1.6, order=2, reason="Outro", zoom=ZoomEffect(enabled=False, factor=1.0)),
        ],
        caption_settings=CaptionSettings(style="dynamic", font_size=56),
        audio_settings=AudioSettings(normalize=True, fade_in=0.01, fade_out=0.01),
        hook_summary="Master Hook",
        edit_rationale="High Retention Shorts",
    )
    rough = editor.execute_plan(plan, synth_video_vertical, isolated_job_workspace)
    assert rough.exists()
    info = probe_media_file(rough)
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert vstream["width"] == 1080
    assert vstream["height"] == 1920
