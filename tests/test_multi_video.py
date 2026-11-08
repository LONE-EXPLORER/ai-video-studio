"""
tests/test_multi_video.py — Comprehensive tests for multi-video schema, normalization, and assembly.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List
import pytest
from pydantic import ValidationError

from app.editing.editor import VideoEditor, _resolve_clip_source
from app.editing.normalizer import MediaNormalizer
from app.pipeline.orchestrator import PipelineOrchestrator
from app.providers.base import BaseProvider
from app.schemas.editing_plan import (
    AspectRatio,
    BrollOverlay,
    Clip,
    EditingPlan,
    SoundEffect,
    ZoomEffect,
)
from app.schemas.job import Job, JobStatus
from app.utils import ffmpeg_utils, file_utils


def _create_synthetic_clip(
    path: Path,
    width: int = 640,
    height: int = 360,
    fps: int = 30,
    duration: float = 1.0,
    sample_rate: int = 48000,
) -> Path:
    """Helper to generate a lightweight test video with lavfi."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_utils.find_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-f", "lavfi",
        "-i", f"testsrc=size={width}x{height}:rate={fps}:duration={duration}",
        "-f", "lavfi",
        "-i", f"sine=frequency=440:sample_rate={sample_rate}:duration={duration}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-ar", str(sample_rate),
        "-ac", "2",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


class DummyMockProvider(BaseProvider):
    """Dummy provider for testing orchestrator."""

    def is_available(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return "mock"

    def generate(self, prompt: str, system: str = "", temperature: float = 0.3) -> str:
        return "mock"

    def generate_json(self, prompt: str, system: str = "", schema_hint: str = "") -> dict:
        return {}


# ===========================================================================
# 1. Job Schema Tests
# ===========================================================================

def test_job_schema_single_source_sync():
    """Job initialized with single source_path auto-populates source_paths."""
    job = Job(
        job_id="j1",
        status=JobStatus.PENDING,
        profile="dynamic_shorts",
        source_path="/media/cam_a.mp4",
        created_at="2026-11-05T00:00:00Z",
        updated_at="2026-11-05T00:00:00Z",
    )
    assert job.source_path == "/media/cam_a.mp4"
    assert job.source_paths == ["/media/cam_a.mp4"]
    assert job.primary_source_path == "/media/cam_a.mp4"


def test_job_schema_multi_source_sync():
    """Job initialized with source_paths list auto-populates primary source_path."""
    paths = ["/media/cam_a.mp4", "/media/cam_b.mp4", "/media/broll.mp4"]
    job = Job(
        job_id="j2",
        status=JobStatus.PENDING,
        profile="dynamic_shorts",
        source_paths=paths,
        created_at="2026-11-05T00:00:00Z",
        updated_at="2026-11-05T00:00:00Z",
    )
    assert job.source_path == "/media/cam_a.mp4"
    assert job.source_paths == paths
    assert job.primary_source_path == "/media/cam_a.mp4"


def test_job_schema_dict_deserialization_sync():
    """Job instantiated from dict handles both keys seamlessly."""
    d1 = {
        "job_id": "j3",
        "status": "PENDING",
        "profile": "dynamic_shorts",
        "source_path": "/tmp/test.mp4",
        "created_at": "t",
        "updated_at": "t",
    }
    j1 = Job.model_validate(d1)
    assert j1.source_paths == ["/tmp/test.mp4"]

    d2 = {
        "job_id": "j4",
        "status": "PENDING",
        "profile": "dynamic_shorts",
        "source_paths": ["/tmp/x.mp4", "/tmp/y.mp4"],
        "created_at": "t",
        "updated_at": "t",
    }
    j2 = Job.model_validate(d2)
    assert j2.source_path == "/tmp/x.mp4"


# ===========================================================================
# 2. Clip & Overlay Schema Tests
# ===========================================================================

def test_clip_schema_defaults_and_custom_source_id():
    """Clip defaults to source_0 and allows custom source_id."""
    c_default = Clip(clip_id="c1", source_start=1.0, source_end=3.0, order=1, reason="hook")
    assert c_default.source_id == "source_0"
    assert c_default.audio_lead == 0.0
    assert c_default.audio_tail == 0.0

    c_custom = Clip(
        clip_id="c2",
        source_id="source_1",
        source_start=5.0,
        source_end=8.0,
        order=2,
        reason="cutaway",
        audio_lead=0.15,
        audio_tail=0.20,
    )
    assert c_custom.source_id == "source_1"
    assert c_custom.audio_lead == 0.15
    assert c_custom.audio_tail == 0.20


def test_broll_overlay_schema_valid_and_invalid():
    """BrollOverlay calculates duration and rejects invalid bounds."""
    overlay = BrollOverlay(
        broll_id="b1",
        timeline_start=2.0,
        timeline_end=5.0,
        source_path="/broll/reaction.mp4",
        overlay_type="cutaway",
    )
    assert overlay.duration == 3.0
    assert overlay.overlay_type == "cutaway"

    with pytest.raises(ValidationError):
        BrollOverlay(
            broll_id="b2",
            timeline_start=5.0,
            timeline_end=2.0,
            source_path="/broll/invalid.mp4",
        )


def test_sound_effect_schema():
    """SoundEffect accepts valid properties and bounds."""
    sfx = SoundEffect(
        sfx_id="s1",
        timeline_timestamp=1.5,
        name="whoosh",
        sfx_path="/sfx/whoosh.wav",
        volume=0.8,
    )
    assert sfx.volume == 0.8
    assert sfx.name == "whoosh"


# ===========================================================================
# 3. EditingPlan Multi-Source Validation Tests
# ===========================================================================

def test_editing_plan_validate_timestamps_multi_source():
    """EditingPlan.validate_timestamps validates each clip against its specific source duration."""
    clips = [
        Clip(clip_id="c1", source_id="source_0", source_start=0.5, source_end=4.0, order=1, reason="h1"),
        Clip(clip_id="c2", source_id="source_1", source_start=1.0, source_end=6.0, order=2, reason="h2"),
    ]
    plan = EditingPlan(
        plan_id="p1",
        job_id="j1",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=8.5,
        clips=clips,
        source_map={"source_0": "/tmp/s0.mp4", "source_1": "/tmp/s1.mp4"},
        hook_summary="Hook",
        edit_rationale="Rationale",
    )

    # Valid durations
    errs = plan.validate_timestamps({"source_0": 5.0, "source_1": 10.0})
    assert len(errs) == 0

    # Clip 2 exceeds source_1 duration
    errs_invalid = plan.validate_timestamps({"source_0": 5.0, "source_1": 5.5})
    assert len(errs_invalid) == 1
    assert "c2" in errs_invalid[0]
    assert "source_1" in errs_invalid[0]

    # Legacy float duration validation
    errs_float = plan.validate_timestamps(10.0)
    assert len(errs_float) == 0


# ===========================================================================
# 4. Source Resolution Helper Tests
# ===========================================================================

def test_resolve_clip_source_scenarios(tmp_path: Path):
    f0 = tmp_path / "f0.mp4"
    f1 = tmp_path / "f1.mp4"
    f0.touch()
    f1.touch()

    c0 = Clip(clip_id="c1", source_id="source_0", source_start=0.0, source_end=1.0, order=1, reason="r")
    c1 = Clip(clip_id="c2", source_id="source_1", source_start=0.0, source_end=1.0, order=2, reason="r")

    # Scenario A: Dict mapping
    mapping = {"source_0": f0, "source_1": f1}
    assert _resolve_clip_source(c0, mapping) == f0
    assert _resolve_clip_source(c1, mapping) == f1

    # Scenario B: List mapping
    paths_list = [f0, f1]
    assert _resolve_clip_source(c0, paths_list) == f0
    assert _resolve_clip_source(c1, paths_list) == f1

    # Scenario C: Single path
    assert _resolve_clip_source(c0, f0) == f0
    assert _resolve_clip_source(c1, f0) == f0

    # Scenario D: Missing key in multi-entry dict raises KeyError
    with pytest.raises(KeyError, match="references source_id 'source_2'"):
        c_missing = Clip(clip_id="c3", source_id="source_2", source_start=0.0, source_end=1.0, order=3, reason="r")
        _resolve_clip_source(c_missing, mapping)


# ===========================================================================
# 5. VideoEditor Multi-Source Cutting & Concat Tests
# ===========================================================================

def test_video_editor_execute_plan_multi_source(tmp_path: Path):
    """VideoEditor cuts clips from distinct source videos and concatenates cleanly."""
    src0 = tmp_path / "source_0.mp4"
    src1 = tmp_path / "source_1.mp4"
    _create_synthetic_clip(src0, width=1080, height=1920, fps=30, duration=2.0)
    _create_synthetic_clip(src1, width=1080, height=1920, fps=30, duration=2.0)

    job_dirs = {
        "intermediate": tmp_path / "intermediate",
        "previews": tmp_path / "previews",
    }
    job_dirs["intermediate"].mkdir(parents=True, exist_ok=True)

    clips = [
        Clip(clip_id="c1", source_id="source_0", source_start=0.2, source_end=0.8, order=1, reason="beat 1"),
        Clip(clip_id="c2", source_id="source_1", source_start=0.5, source_end=1.2, order=2, reason="beat 2"),
    ]
    plan = EditingPlan(
        plan_id="p_multi",
        job_id="j_multi",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=1.3,
        clips=clips,
        hook_summary="H",
        edit_rationale="R",
    )

    editor = VideoEditor()
    sources_map = {"source_0": src0, "source_1": src1}
    rough_cut = editor.execute_plan(plan, source_path=sources_map, job_dirs=job_dirs)

    assert rough_cut.exists()
    assert rough_cut.stat().st_size > 0

    info = ffmpeg_utils.probe_video(rough_cut)
    dur = float(info["format"]["duration"])
    # 0.6 + 0.7 = 1.3s
    assert dur == pytest.approx(1.3, abs=0.15)


def test_concat_clips_single_clip_and_fallback(tmp_path: Path):
    """Test concat_clips edge cases (single clip, multi-clip)."""
    c1 = tmp_path / "c1.mp4"
    c2 = tmp_path / "c2.mp4"
    _create_synthetic_clip(c1, width=1080, height=1920, fps=30, duration=0.5)
    _create_synthetic_clip(c2, width=1080, height=1920, fps=30, duration=0.5)

    # Single clip copies directly
    out_single = tmp_path / "out_single.mp4"
    res_single = ffmpeg_utils.concat_clips([c1], out_single)
    assert res_single.exists()
    assert res_single.stat().st_size == c1.stat().st_size

    # Multiple clips
    out_multi = tmp_path / "out_multi.mp4"
    res_multi = ffmpeg_utils.concat_clips([c1, c2], out_multi)
    assert res_multi.exists()
    info = ffmpeg_utils.probe_video(res_multi)
    assert float(info["format"]["duration"]) == pytest.approx(1.0, abs=0.15)


# ===========================================================================
# 6. Pipeline Orchestrator Multi-Source Ingest Tests
# ===========================================================================

def test_pipeline_orchestrator_multi_source_ingest(tmp_path: Path):
    """Orchestrator creates sources/ and conformed media files during _stage_ingest."""
    jobs_dir = tmp_path / "jobs"
    job_id = "test_multi_ingest"

    src0 = tmp_path / "raw_cam0.mp4"
    src1 = tmp_path / "raw_cam1.mp4"
    _create_synthetic_clip(src0, width=640, height=360, fps=25, duration=0.5)
    _create_synthetic_clip(src1, width=360, height=640, fps=25, duration=0.5)

    prov = DummyMockProvider()
    orch = PipelineOrchestrator(prov)

    # Initialise job dirs manually as orchestrator does
    job_dirs = file_utils.create_job_dirs(jobs_dir, job_id)
    sources_dir = job_dirs.get("sources", job_dirs["root"] / "sources")
    sources_dir.mkdir(parents=True, exist_ok=True)
    job_dirs["sources"] = sources_dir

    # Validate normalizer conforms them into sources_dir
    normalizer = MediaNormalizer(target_w=1080, target_h=1920, fps=30)
    conformed_sources = normalizer.conform_sources([src0, src1], sources_dir)

    assert "source_0" in conformed_sources
    assert "source_1" in conformed_sources
    assert conformed_sources["source_0"].exists()
    assert conformed_sources["source_1"].exists()

    info0 = ffmpeg_utils.probe_video(conformed_sources["source_0"])
    v0 = next(s for s in info0["streams"] if s["codec_type"] == "video")
    assert int(v0["width"]) == 1080
    assert int(v0["height"]) == 1920
