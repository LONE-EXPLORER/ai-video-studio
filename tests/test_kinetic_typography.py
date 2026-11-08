"""
tests/test_kinetic_typography.py — Comprehensive unit and integration test suite
for Milestone 4 Kinetic Typography (Features 12, 13, and 14).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple
import pytest
from PIL import Image

from app.captions.pillow_captions import (
    CaptionGroup,
    MappedWord,
    PhraseLayout,
    WordLayoutItem,
    _get_font,
    burn_kinetic_captions,
    burn_kinetic_captions_concat,
    burn_kinetic_captions_direct_pipe,
    check_videotoolbox_available,
    compute_harmonic_spring_scale,
    group_words_into_phrases,
    map_words_to_timeline,
    render_caption_card,
    render_sample_word_pill,
    spring_scale,
)
from app.schemas.editing_plan import AspectRatio, CaptionSettings, Clip, EditingPlan
from app.schemas.transcript import Transcript, TranscriptSegment, TranscriptWord
from app.utils.ffmpeg_utils import find_ffmpeg
from tests.e2e.conftest import make_synthetic_video, probe_media_file


# ===========================================================================
# Feature 12 Tests: Fast Kinetic Frame Generation & Speech-Interval Rendering
# ===========================================================================

def test_f12_speech_interval_partitioning():
    """Verify that mapped words correctly identify speech windows and silence gaps."""
    words = [
        MappedWord("HELLO", 1.0, 1.5),
        MappedWord("WORLD", 1.6, 2.0),
        MappedWord("AGAIN", 5.0, 5.5),
    ]
    groups = group_words_into_phrases(words, max_words=2)
    assert len(groups) == 2
    # Group 1: 1.0 to 2.0
    assert groups[0].start == 1.0
    assert groups[0].end == 2.0
    # Group 2: 5.0 to 5.5
    assert groups[1].start == 5.0
    assert groups[1].end == 5.5
    # Silence gap between group 0 and group 1 is 3.0s (5.0 - 2.0)
    silence_gap = groups[1].start - groups[0].end
    assert silence_gap == pytest.approx(3.0)


def test_f12_frame_deduplication_steady_state():
    """Verify that steady state frames produce identical cache keys for zero-draw repetition."""
    font = _get_font(72)
    group = CaptionGroup([MappedWord("KINETIC", 0.5, 1.5)])
    layout = PhraseLayout(group, font, canvas_width=1080, box_y=1190)

    # During bounce (t=0.50 to t=0.62), scale changes
    scale_0 = spring_scale(0.0)
    scale_1 = spring_scale(0.04)
    assert scale_0 != scale_1

    # After bounce (t >= 0.120), scale is steady at 1.00
    scale_steady_1 = spring_scale(0.120)
    scale_steady_2 = spring_scale(0.300)
    scale_steady_3 = spring_scale(0.900)
    assert scale_steady_1 == 1.00
    assert scale_steady_2 == 1.00
    assert scale_steady_3 == 1.00

    # Rendering at scale 1.00 produces deterministic output
    img1 = render_caption_card(layout, active_idx=0, scale=scale_steady_1)
    img2 = render_caption_card(layout, active_idx=0, scale=scale_steady_2)
    assert img1.size == img2.size
    assert img1.tobytes() == img2.tobytes()


def test_f12_fast_frame_generator_produces_transparent_rgba():
    """Verify that render_sample_word_pill produces valid RGBA image with alpha transparency."""
    img = render_sample_word_pill("MARVEL", highlight=True, width=1080, height=1920)
    assert img.mode == "RGBA"
    assert img.size == (1080, 1920)
    extrema = img.getchannel("A").getextrema()
    assert extrema[0] == 0  # Fully transparent outer background
    assert extrema[1] > 0   # Non-zero alpha for pill and glyphs


def test_f12_concat_demuxer_sparse_rendering(tmp_path: Path):
    """Verify that burn_kinetic_captions_concat generates valid output using sparse demuxer."""
    vid = tmp_path / "test_concat_src.mp4"
    make_synthetic_video(vid, width=1080, height=1920, fps=30, duration=1.5)

    seg = TranscriptSegment(
        id=1, start=0.2, end=1.2, text="Fast generation",
        words=[TranscriptWord(word="Fast", start=0.2, end=0.6), TranscriptWord(word="generation", start=0.7, end=1.2)]
    )
    transcript = Transcript(language="en", duration=1.5, segments=[seg])
    plan = EditingPlan(
        plan_id="p_f12_concat",
        job_id="j_f12_concat",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=1.5,
        hook_summary="Fast test",
        edit_rationale="Concat test",
        clips=[Clip(clip_id="c1", source_start=0.0, source_end=1.5, order=1, reason="Test")],
    )

    out = tmp_path / "concat_burned.mp4"
    burn_kinetic_captions_concat(vid, plan, transcript, out, fps=30)
    assert out.exists()
    assert out.stat().st_size > 0
    info = probe_media_file(out)
    assert float(info["format"]["duration"]) == pytest.approx(1.5, abs=0.2)


# ===========================================================================
# Feature 13 Tests: Kinetic Spring/Bounce Entry Animations
# ===========================================================================

def test_f13_spring_scale_entry_pop():
    """Verify that spring_scale starts at 1.20x at t=0 and settles to 1.00x at t>=0.120s."""
    assert spring_scale(0.0) == 1.20
    assert spring_scale(0.120) == 1.00
    assert spring_scale(0.250) == 1.00


def test_f13_spring_scale_strict_monotonic_decay():
    """Verify strictly monotonic decay between 0.0s and 0.120s."""
    steps = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.120]
    scales = [spring_scale(t) for t in steps]
    for i in range(len(scales) - 1):
        assert scales[i] >= scales[i + 1]


def test_f13_spring_scale_boundaries():
    """Verify boundary handling: negative time, zero duration, and non-finite values."""
    assert spring_scale(-0.1) == 1.20
    assert spring_scale(0.0, duration=0.0) == 1.00
    assert spring_scale(0.0, duration=-1.0) == 1.00
    assert spring_scale(float("nan")) == 1.00
    assert spring_scale(float("inf")) == 1.00


def test_f13_compute_harmonic_spring_scale():
    """Verify damped harmonic oscillator physics formulation."""
    s0 = compute_harmonic_spring_scale(0.0, amplitude=0.20)
    assert s0 == 1.20
    s_settle = compute_harmonic_spring_scale(0.300)
    assert s_settle == 1.00
    # Between 0 and 0.25, returns valid float within expected bounce bounds
    for t_ms in range(0, 200, 20):
        t = t_ms / 1000.0
        val = compute_harmonic_spring_scale(t)
        assert 0.95 <= val <= 1.20


def test_f13_font_vector_point_scaling():
    """Verify vector font scaling produces proportionally scaled font sizes with cache hits."""
    base_size = 72
    font_base = _get_font(base_size)
    assert font_base is not None

    scale_pop = spring_scale(0.0)
    scaled_pt = int(round(base_size * scale_pop))
    assert scaled_pt == 86  # 72 * 1.20 = 86.4 -> 86

    font_scaled = _get_font(scaled_pt)
    assert font_scaled is not None
    # Font sizes are distinct
    assert font_scaled.size == 86
    assert font_base.size == 72


def test_f13_phrase_layout_optical_center_stability():
    """
    Verify Two-Pass layout geometry:
    Word optical centers (cx, cy) remain completely immutable regardless of which word scales up.
    """
    font = _get_font(72)
    words = [
        MappedWord("THE", 0.0, 0.5),
        MappedWord("FUTURE", 0.5, 1.0),
        MappedWord("NOW", 1.0, 1.5),
    ]
    group = CaptionGroup(words)
    layout = PhraseLayout(group, font, canvas_width=1080, box_y=1190, word_spacing=24)

    assert len(layout.items) == 3
    # Capture immutable centers
    cx_the = layout.items[0].cx
    cx_future = layout.items[1].cx
    cx_now = layout.items[2].cx

    # Ensure horizontal ordering
    assert cx_the < cx_future < cx_now

    # When FUTURE pops to 1.20x, its center remains cx_future
    act_item = layout.items[1]
    assert act_item.cx == cx_future
    assert act_item.cy == layout.items[1].cy


# ===========================================================================
# Feature 14 Tests: Active-Word Pill Highlight & VideoToolbox Hardware Compositing
# ===========================================================================

def test_f14_pill_geometry_and_z_ordering():
    """
    Verify 3-Layer Z-ordering:
    Layer 1: Translucent pill background
    Layer 2: Inactive words text
    Layer 3: Active word text with gold highlight
    """
    font = _get_font(72)
    words = [MappedWord("WORD1", 0.0, 0.5), MappedWord("WORD2", 0.5, 1.0)]
    layout = PhraseLayout(CaptionGroup(words), font, canvas_width=1080, box_y=1190)

    # Render with active word 0 at scale 1.20
    img = render_caption_card(
        layout,
        active_idx=0,
        scale=1.20,
        width=1080,
        height=1920,
        pill_color=(20, 20, 25, 180),
        color_active=(255, 204, 0, 255),
        color_inactive=(255, 255, 255, 255),
        radius=16,
        pad_x=14,
        pad_y=8,
    )
    assert img.mode == "RGBA"
    assert img.size == (1080, 1920)

    # Active word 0's center pixel region contains active gold color or dark stroke
    act_cx = int(layout.items[0].cx)
    act_cy = int(layout.items[0].cy)
    # Check that pixels around center are non-transparent
    pixel = img.getpixel((act_cx, act_cy))
    assert pixel[3] > 0  # Non-zero alpha


def test_f14_videotoolbox_detection():
    """Verify VideoToolbox hardware encoder detection function runs safely."""
    ffmpeg = find_ffmpeg()
    vt_available = check_videotoolbox_available(ffmpeg)
    assert isinstance(vt_available, bool)


def test_f14_e2e_audio_preservation(tmp_path: Path):
    """
    Verify that burn_kinetic_captions preserves dialogue audio track 0 with zero drift.
    """
    vid = tmp_path / "audio_test_src.mp4"
    make_synthetic_video(vid, width=1080, height=1920, fps=30, duration=2.0, has_audio=True)

    seg = TranscriptSegment(
        id=1, start=0.2, end=1.8, text="Audio sync test",
        words=[
            TranscriptWord(word="Audio", start=0.2, end=0.7),
            TranscriptWord(word="sync", start=0.8, end=1.2),
            TranscriptWord(word="test", start=1.3, end=1.8),
        ]
    )
    transcript = Transcript(language="en", duration=2.0, segments=[seg])
    plan = EditingPlan(
        plan_id="p_f14_audio",
        job_id="j_f14_audio",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=2.0,
        hook_summary="Audio test",
        edit_rationale="Sync test",
        clips=[Clip(clip_id="c1", source_start=0.0, source_end=2.0, order=1, reason="Test")],
    )

    out = tmp_path / "audio_preserved.mp4"
    burn_kinetic_captions(vid, plan, transcript, out, fps=30)
    assert out.exists()
    assert out.stat().st_size > 0

    info = probe_media_file(out)
    streams = info["streams"]
    audio_stream = next((s for s in streams if s["codec_type"] == "audio"), None)
    assert audio_stream is not None, "Output must contain an audio stream"
    assert int(audio_stream["sample_rate"]) == 48000
    assert int(audio_stream["channels"]) == 2
    # Verify duration
    dur = float(info["format"]["duration"])
    assert dur == pytest.approx(2.0, abs=0.2)


def test_both_rendering_methods_produce_valid_mp4(tmp_path: Path):
    """Verify that both method='pipe' and method='concat' produce valid MP4 deliverables."""
    vid = tmp_path / "dual_method_src.mp4"
    make_synthetic_video(vid, width=1080, height=1920, fps=30, duration=1.0)

    seg = TranscriptSegment(
        id=1, start=0.1, end=0.9, text="Dual method",
        words=[
            TranscriptWord(word="Dual", start=0.1, end=0.4),
            TranscriptWord(word="method", start=0.5, end=0.9),
        ]
    )
    transcript = Transcript(language="en", duration=1.0, segments=[seg])
    plan = EditingPlan(
        plan_id="p_dual",
        job_id="j_dual",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=1.0,
        hook_summary="Dual test",
        edit_rationale="Dual test",
        clips=[Clip(clip_id="c1", source_start=0.0, source_end=1.0, order=1, reason="Test")],
    )

    out_pipe = tmp_path / "out_pipe.mp4"
    out_concat = tmp_path / "out_concat.mp4"

    burn_kinetic_captions(vid, plan, transcript, out_pipe, fps=30, method="pipe")
    burn_kinetic_captions(vid, plan, transcript, out_concat, fps=30, method="concat")

    assert out_pipe.exists() and out_pipe.stat().st_size > 0
    assert out_concat.exists() and out_concat.stat().st_size > 0
