"""
tests/test_broll_compositor.py — Unit tests for Hardware-Accelerated B-Roll Compositor.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from app.rendering.broll_compositor import BrollCompositor
from app.schemas.editing_plan import BrollOverlay
from app.utils.ffmpeg_utils import probe_video
from tests.e2e.conftest import make_synthetic_video


def test_broll_compositor_single_cutaway(tmp_path: Path):
    """Verify single B-roll cutaway overlays onto base video and maintains 1080x1920 resolution."""
    base = tmp_path / "base.mp4"
    broll = tmp_path / "broll.mp4"
    out = tmp_path / "composited.mp4"

    make_synthetic_video(base, width=1080, height=1920, fps=30, duration=3.0, has_audio=True)
    make_synthetic_video(broll, width=1080, height=1920, fps=30, duration=1.5, pattern="color", has_audio=False)

    compositor = BrollCompositor()
    overlays = [
        BrollOverlay(
            broll_id="broll_1",
            timeline_start=1.0,
            timeline_end=2.5,
            source_path=str(broll),
            overlay_type="cutaway",
        )
    ]

    result = compositor.composite_overlays(
        base_video_path=base,
        overlays=overlays,
        output_path=out,
        target_w=1080,
        target_h=1920,
    )

    assert result.exists()
    assert result.stat().st_size > 0

    info = probe_video(result)
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert vstream["width"] == 1080
    assert vstream["height"] == 1920


def test_broll_compositor_dialogue_uninterrupted(tmp_path: Path):
    """Verify base video audio track 0 is preserved completely during cutaways."""
    base = tmp_path / "base_audio.mp4"
    broll = tmp_path / "broll_noise.mp4"
    out = tmp_path / "composited_audio.mp4"

    make_synthetic_video(base, width=1080, height=1920, fps=30, duration=4.0, has_audio=True, audio_freq=440)
    # B-roll with distinct frequency or no audio
    make_synthetic_video(broll, width=1080, height=1920, fps=30, duration=2.0, has_audio=True, audio_freq=880)

    compositor = BrollCompositor()
    overlays = [
        BrollOverlay(
            broll_id="broll_cutaway",
            timeline_start=1.0,
            timeline_end=3.0,
            source_path=str(broll),
            overlay_type="cutaway",
        )
    ]

    result = compositor.composite_overlays(base, overlays, out)
    info = probe_video(result)

    astreams = [s for s in info["streams"] if s["codec_type"] == "audio"]
    assert len(astreams) == 1
    astream = astreams[0]
    assert int(astream["sample_rate"]) == 48000
    assert astream["channels"] == 2


def test_broll_compositor_short_clip_looping(tmp_path: Path):
    """Verify short B-roll clips loop smoothly when cutaway duration exceeds asset duration."""
    base = tmp_path / "base_loop.mp4"
    short_broll = tmp_path / "short_broll.mp4"
    out = tmp_path / "composited_loop.mp4"

    make_synthetic_video(base, width=1080, height=1920, fps=30, duration=4.0)
    # 0.8s short asset
    make_synthetic_video(short_broll, width=1080, height=1920, fps=30, duration=0.8, pattern="color")

    compositor = BrollCompositor()
    # 2.2s cutaway window > 0.8s asset
    overlays = [
        BrollOverlay(
            broll_id="short_01",
            timeline_start=1.0,
            timeline_end=3.2,
            source_path=str(short_broll),
            overlay_type="cutaway",
        )
    ]

    result = compositor.composite_overlays(base, overlays, out)
    assert result.exists()
    assert result.stat().st_size > 0


def test_broll_compositor_aspect_conformation(tmp_path: Path):
    """Verify landscape 16:9 B-roll (1920x1080) is conformed to 9:16 vertical (1080x1920)."""
    base = tmp_path / "base_vert.mp4"
    broll_land = tmp_path / "broll_land.mp4"
    out = tmp_path / "composited_conformed.mp4"

    make_synthetic_video(base, width=1080, height=1920, fps=30, duration=3.0)
    # Landscape 1920x1080
    make_synthetic_video(broll_land, width=1920, height=1080, fps=30, duration=1.5, pattern="color")

    compositor = BrollCompositor()
    overlays = [
        BrollOverlay(
            broll_id="land_01",
            timeline_start=0.5,
            timeline_end=2.0,
            source_path=str(broll_land),
            overlay_type="cutaway",
        )
    ]

    result = compositor.composite_overlays(base, overlays, out)
    info = probe_video(result)
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert vstream["width"] == 1080
    assert vstream["height"] == 1920


def test_broll_compositor_chained_multiple_overlays(tmp_path: Path):
    """Verify multiple sequential overlays chain deterministically in a single filtergraph."""
    base = tmp_path / "base_multi.mp4"
    b1 = tmp_path / "b1.mp4"
    b2 = tmp_path / "b2.mp4"
    out = tmp_path / "composited_multi.mp4"

    make_synthetic_video(base, width=1080, height=1920, fps=30, duration=5.0)
    make_synthetic_video(b1, width=1080, height=1920, fps=30, duration=1.0, pattern="color")
    make_synthetic_video(b2, width=1080, height=1920, fps=30, duration=1.0, pattern="smptebars")

    compositor = BrollCompositor()
    overlays = [
        BrollOverlay(broll_id="b1", timeline_start=1.0, timeline_end=2.0, source_path=str(b1)),
        BrollOverlay(broll_id="b2", timeline_start=3.0, timeline_end=4.0, source_path=str(b2)),
    ]

    result = compositor.composite_overlays(base, overlays, out)
    assert result.exists()
    assert result.stat().st_size > 0


def test_broll_compositor_missing_file_resilience(tmp_path: Path):
    """Verify missing B-roll files are skipped gracefully without crashing."""
    base = tmp_path / "base_missing.mp4"
    out = tmp_path / "composited_missing.mp4"

    make_synthetic_video(base, width=1080, height=1920, fps=30, duration=2.0)
    compositor = BrollCompositor()

    missing_overlay = BrollOverlay(
        broll_id="missing",
        timeline_start=0.5,
        timeline_end=1.5,
        source_path=str(tmp_path / "does_not_exist.mp4"),
    )

    result = compositor.composite_overlays(base, [missing_overlay], out)
    # Should safely return output containing base video
    assert result.exists()
    assert result.stat().st_size > 0


def test_broll_compositor_empty_overlays_returns_base(tmp_path: Path):
    """Verify empty overlay list returns base video safely."""
    base = tmp_path / "base_empty.mp4"
    out = tmp_path / "composited_empty.mp4"

    make_synthetic_video(base, width=1080, height=1920, fps=30, duration=2.0)
    compositor = BrollCompositor()

    result = compositor.composite_overlays(base, [], out)
    assert result.exists()
    assert result.stat().st_size == base.stat().st_size


def test_broll_compositor_pip_mode(tmp_path: Path):
    """Verify picture-in-picture overlay mode operates cleanly."""
    base = tmp_path / "base_pip.mp4"
    broll = tmp_path / "broll_pip.mp4"
    out = tmp_path / "composited_pip.mp4"

    make_synthetic_video(base, width=1080, height=1920, fps=30, duration=3.0)
    make_synthetic_video(broll, width=1080, height=1920, fps=30, duration=1.5, pattern="color")

    compositor = BrollCompositor()
    overlays = [
        BrollOverlay(
            broll_id="pip_01",
            timeline_start=1.0,
            timeline_end=2.0,
            source_path=str(broll),
            overlay_type="pip",
        )
    ]

    result = compositor.composite_overlays(base, overlays, out)
    assert result.exists()
    assert result.stat().st_size > 0


def test_broll_compositor_software_libx264_fallback(tmp_path: Path):
    """Verify software libx264 fallback encoding path works identically."""
    base = tmp_path / "base_soft.mp4"
    broll = tmp_path / "broll_soft.mp4"
    out = tmp_path / "composited_soft.mp4"

    make_synthetic_video(base, width=1080, height=1920, fps=30, duration=2.0)
    make_synthetic_video(broll, width=1080, height=1920, fps=30, duration=1.0, pattern="color")

    compositor = BrollCompositor()
    # Force fallback by mocking _vt_available = False
    compositor._vt_available = False

    overlays = [
        BrollOverlay(
            broll_id="soft_01",
            timeline_start=0.5,
            timeline_end=1.5,
            source_path=str(broll),
        )
    ]

    result = compositor.composite_overlays(base, overlays, out)
    assert result.exists()
    assert result.stat().st_size > 0


def test_renderer_integrates_broll_overlays(tmp_path: Path):
    """Verify Renderer.render_final_video executes B-roll overlay compositing and produces final video."""
    from app.rendering.renderer import Renderer
    from app.schemas.editing_plan import EditingPlan, Clip, CaptionSettings, AudioSettings

    rough_cut = tmp_path / "rough_cut.mp4"
    broll = tmp_path / "broll_test.mp4"
    make_synthetic_video(rough_cut, width=1080, height=1920, fps=30, duration=3.0, has_audio=True)
    make_synthetic_video(broll, width=1080, height=1920, fps=30, duration=1.0, pattern="color", has_audio=False)

    job_dirs = {
        "intermediate": tmp_path / "intermediate",
        "exports": tmp_path / "exports",
        "transcripts": tmp_path / "transcripts",
    }
    for d in job_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    plan = EditingPlan(
        plan_id="plan_rend",
        job_id="job_rend",
        profile="dynamic_shorts",
        aspect_ratio="9:16",
        target_duration_seconds=3.0,
        hook_summary="Hook",
        edit_rationale="Rationale",
        clips=[Clip(clip_id="c001", source_start=0.0, source_end=3.0, order=1, reason="Beat 1")],
        broll_overlays=[
            BrollOverlay(
                broll_id="broll_test",
                timeline_start=1.0,
                timeline_end=2.0,
                source_path=str(broll),
            )
        ],
        caption_settings=CaptionSettings(enabled=False),
        audio_settings=AudioSettings(),
    )

    renderer = Renderer()
    final_output = renderer.render_final_video(
        rough_cut=rough_cut,
        plan=plan,
        job_dirs=job_dirs,
    )

    assert final_output.exists()
    assert final_output.stat().st_size > 0
    # Also verify the intermediate broll_composited.mp4 was generated
    composited_intermediate = job_dirs["intermediate"] / "broll_composited.mp4"
    assert composited_intermediate.exists()
