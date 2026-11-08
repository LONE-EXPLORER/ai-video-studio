"""
tests/test_adversarial_m1.py — Empirical Challenger stress-test harness for Milestone 1.

Covers:
1. Pydantic schema validation under adversarial inputs:
   - Malformed types, invalid enums, non-string arrays
   - Corrupted JSON (syntax errors, empty payloads, null values)
   - Negative timestamps, inverted bounds, zero durations
   - Infinite / NaN floating point values in timestamps and audio offsets
   - SFX and B-Roll boundary stress-tests
   - EditingPlan.validate_timestamps under adversarial source maps and unknown source IDs
2. PipelineOrchestrator _stage_ingest with multi-source arrays (1, 2, 5 files):
   - Conforming heterogeneous formats (aspect ratio, fps, audio presence)
   - Basename collision testing (same filename in different folders)
   - Non-existent files, 0-byte files, and empty array handling
"""

from __future__ import annotations

import json
import math
import subprocess
import uuid
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch

import pytest
from pydantic import ValidationError

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
from app.schemas.job import Job, JobStatus, StageProgress
from app.utils import db, ffmpeg_utils, file_utils


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

class MockProvider(BaseProvider):
    def is_available(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return "mock"

    def generate(self, prompt: str, system: str = "", temperature: float = 0.3) -> str:
        return "mock"

    def generate_json(self, prompt: str, system: str = "", schema_hint: str = "") -> dict:
        return {}


def create_synthetic_test_video(
    path: Path,
    width: int = 640,
    height: int = 360,
    fps: int = 30,
    duration: float = 1.0,
    sample_rate: int = 48000,
    has_audio: bool = True,
) -> Path:
    """Generate lightweight synthetic video for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = ffmpeg_utils.find_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-f", "lavfi", "-i", f"testsrc=size={width}x{height}:rate={fps}:duration={duration}",
    ]
    if has_audio:
        cmd += [
            "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate={sample_rate}:duration={duration}",
            "-c:a", "aac", "-ar", str(sample_rate), "-ac", "2",
        ]
    else:
        cmd += ["-an"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


# ---------------------------------------------------------------------------
# 1. Pydantic Schema Validation Under Adversarial Inputs
# ---------------------------------------------------------------------------

def test_job_schema_adversarial_corrupted_json():
    """Job schema rejects corrupted JSON payloads and malformed structures."""
    malformed_jsons = [
        "",
        "{",
        '{"job_id": "j1", }',  # Trailing comma syntax error
        "null",
        "[]",
        '{"job_id": null, "status": "PENDING"}',
        '{"job_id": "j1", "status": "INVALID_STATE", "profile": "p", "created_at": "t", "updated_at": "t"}',
    ]
    for bad_json in malformed_jsons:
        with pytest.raises((ValidationError, Exception)):
            Job.model_validate_json(bad_json)


def test_job_schema_adversarial_types():
    """Job schema rejects non-string IDs, invalid statuses, and malformed source_paths."""
    # Invalid status enum
    with pytest.raises(ValidationError):
        Job(
            job_id="j1",
            status="UNKNOWN_STATUS",  # type: ignore
            profile="p",
            created_at="2026-11-05T00:00:00Z",
            updated_at="2026-11-05T00:00:00Z",
        )

    # None inside source_paths
    with pytest.raises(ValidationError):
        Job(
            job_id="j1",
            status=JobStatus.PENDING,
            profile="p",
            source_paths=[None],  # type: ignore
            created_at="t",
            updated_at="t",
        )


def test_clip_schema_negative_and_zero_timestamps():
    """Clip schema strictly rejects negative and zero timestamps."""
    # Negative source_start
    with pytest.raises(ValidationError) as exc:
        Clip(clip_id="c1", source_start=-1.0, source_end=2.0, order=1, reason="r")
    assert "source_start" in str(exc.value)

    # Negative source_end
    with pytest.raises(ValidationError) as exc:
        Clip(clip_id="c1", source_start=1.0, source_end=-2.0, order=1, reason="r")
    assert "source_end" in str(exc.value)

    # Zero duration (start == end)
    with pytest.raises(ValidationError) as exc:
        Clip(clip_id="c1", source_start=3.0, source_end=3.0, order=1, reason="r")
    assert "greater than source_start" in str(exc.value)

    # Inverted timestamps (start > end)
    with pytest.raises(ValidationError) as exc:
        Clip(clip_id="c1", source_start=5.0, source_end=2.0, order=1, reason="r")
    assert "greater than source_start" in str(exc.value)

    # Negative J-cut audio lead
    with pytest.raises(ValidationError) as exc:
        Clip(clip_id="c1", source_start=0.0, source_end=2.0, order=1, reason="r", audio_lead=-0.5)
    assert "audio_lead" in str(exc.value)

    # Negative L-cut audio tail
    with pytest.raises(ValidationError) as exc:
        Clip(clip_id="c1", source_start=0.0, source_end=2.0, order=1, reason="r", audio_tail=-0.5)
    assert "audio_tail" in str(exc.value)


def test_clip_schema_nan_and_inf_values():
    """Clip schema rejects NaN and infinity values."""
    # NaN in start
    with pytest.raises((ValidationError, ValueError)):
        Clip(clip_id="c1", source_start=float("nan"), source_end=2.0, order=1, reason="r")

    # NaN in end
    with pytest.raises((ValidationError, ValueError)):
        Clip(clip_id="c1", source_start=0.0, source_end=float("nan"), order=1, reason="r")

    # Negative infinity
    with pytest.raises((ValidationError, ValueError)):
        Clip(clip_id="c1", source_start=float("-inf"), source_end=2.0, order=1, reason="r")

    # Positive infinity in end
    with pytest.raises((ValidationError, ValueError)) as exc:
        Clip(clip_id="c1", source_start=1.0, source_end=float("inf"), order=1, reason="r")
    assert "finite numbers" in str(exc.value)

    # Positive infinity in start
    with pytest.raises((ValidationError, ValueError)) as exc:
        Clip(clip_id="c2", source_start=float("inf"), source_end=10.0, order=1, reason="r")
    assert "finite numbers" in str(exc.value)

    # Positive infinity in audio_lead
    with pytest.raises((ValidationError, ValueError)) as exc:
        Clip(clip_id="c3", source_start=0.0, source_end=2.0, order=1, reason="r", audio_lead=float("inf"))
    assert "finite numbers" in str(exc.value)

    # Positive infinity in audio_tail
    with pytest.raises((ValidationError, ValueError)) as exc:
        Clip(clip_id="c4", source_start=0.0, source_end=2.0, order=1, reason="r", audio_tail=float("inf"))
    assert "finite numbers" in str(exc.value)


def test_broll_and_sfx_adversarial_bounds():
    """BrollOverlay and SoundEffect reject negative, inverted, and infinite bounds."""
    # BrollOverlay negative start
    with pytest.raises(ValidationError):
        BrollOverlay(broll_id="b1", timeline_start=-1.0, timeline_end=2.0, source_path="b.mp4")

    # BrollOverlay inverted
    with pytest.raises(ValidationError):
        BrollOverlay(broll_id="b1", timeline_start=4.0, timeline_end=2.0, source_path="b.mp4")

    # BrollOverlay infinite
    with pytest.raises(ValidationError) as exc:
        BrollOverlay(broll_id="b1", timeline_start=1.0, timeline_end=float("inf"), source_path="b.mp4")
    assert "finite numbers" in str(exc.value)

    # SoundEffect negative timestamp
    with pytest.raises(ValidationError):
        SoundEffect(sfx_id="s1", timeline_timestamp=-0.5, name="pop", sfx_path="s.wav")

    # SoundEffect infinite timestamp
    with pytest.raises(ValidationError) as exc:
        SoundEffect(sfx_id="s1", timeline_timestamp=float("inf"), name="pop", sfx_path="s.wav")
    assert "finite numbers" in str(exc.value)

    # SoundEffect volume out of bounds
    with pytest.raises(ValidationError):
        SoundEffect(sfx_id="s1", timeline_timestamp=1.0, name="pop", sfx_path="s.wav", volume=1.5)

    with pytest.raises(ValidationError):
        SoundEffect(sfx_id="s1", timeline_timestamp=1.0, name="pop", sfx_path="s.wav", volume=-0.1)


def test_editing_plan_validate_timestamps_adversarial():
    """EditingPlan.validate_timestamps detects invalid source ranges and unknown source IDs."""
    c1 = Clip(clip_id="c1", source_id="source_0", source_start=0.0, source_end=5.0, order=1, reason="r")
    plan = EditingPlan(
        plan_id="p1",
        job_id="j1",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=5.0,
        clips=[c1],
        hook_summary="Hook",
        edit_rationale="Rationale",
    )

    # Valid duration
    assert len(plan.validate_timestamps(10.0)) == 0

    # Clip exceeds source duration
    errs = plan.validate_timestamps(3.0)
    assert len(errs) == 1
    assert "exceeds source duration" in errs[0]

    # Negative source duration
    errs_neg = plan.validate_timestamps(-1.0)
    assert len(errs_neg) == 1

    # Unknown source_id in multi-source dictionary
    clip_unknown = Clip(clip_id="c2", source_id="unknown_source", source_start=0.0, source_end=5.0, order=2, reason="r")
    plan_multi = EditingPlan(
        plan_id="p2",
        job_id="j2",
        profile="dynamic_shorts",
        aspect_ratio=AspectRatio.VERTICAL_9_16,
        target_duration_seconds=5.0,
        clips=[clip_unknown],
        hook_summary="Hook",
        edit_rationale="Rationale",
    )
    errs_unknown = plan_multi.validate_timestamps({"source_0": 10.0})
    assert len(errs_unknown) == 1
    assert "unknown source_id 'unknown_source' not present in source durations" in errs_unknown[0]


# ---------------------------------------------------------------------------
# 2. PipelineOrchestrator _stage_ingest with multi-source arrays (1, 2, 5 files)
# ---------------------------------------------------------------------------

def _run_only_stage_ingest(orch: PipelineOrchestrator, **run_job_kwargs) -> Dict:
    """Helper to execute PipelineOrchestrator up through _stage_ingest."""
    captured_result = {}

    def mock_run_stages(stages, ctx, log_fn):
        ingest_stage = next(s for s in stages if s.name == "ingest")
        res = ingest_stage.fn(ctx)
        captured_result.update(res)
        return res

    with patch("app.pipeline.orchestrator.run_stages", side_effect=mock_run_stages):
        orch.run_job(**run_job_kwargs)
    return captured_result


def test_orchestrator_stage_ingest_1_file(tmp_path: Path):
    """PipelineOrchestrator _stage_ingest with 1 file conforms to 1080x1920 30fps."""
    orch = PipelineOrchestrator(MockProvider())
    orch.config.jobs_dir_raw = str(tmp_path / "jobs")

    src = create_synthetic_test_video(tmp_path / "raw" / "single.mp4", width=640, height=360, fps=25, duration=1.0)
    job_id = str(uuid.uuid4())

    res = _run_only_stage_ingest(orch, job_id=job_id, source_video_paths=[src])

    assert "source_0" in res["conformed_sources"]
    conformed_path = res["conformed_sources"]["source_0"]
    assert conformed_path.exists()

    meta = ffmpeg_utils.probe_video(conformed_path)
    v = next(s for s in meta["streams"] if s["codec_type"] == "video")
    a = next(s for s in meta["streams"] if s["codec_type"] == "audio")
    assert int(v["width"]) == 1080
    assert int(v["height"]) == 1920
    assert int(a["sample_rate"]) == 48000
    assert int(a["channels"]) == 2


def test_orchestrator_stage_ingest_2_files(tmp_path: Path):
    """PipelineOrchestrator _stage_ingest with 2 files conforms both inputs."""
    orch = PipelineOrchestrator(MockProvider())
    orch.config.jobs_dir_raw = str(tmp_path / "jobs")

    src0 = create_synthetic_test_video(tmp_path / "raw" / "cam_a.mp4", width=1920, height=1080, fps=30, duration=1.0)
    src1 = create_synthetic_test_video(tmp_path / "raw" / "cam_b.mp4", width=720, height=1280, fps=24, duration=1.5)
    job_id = str(uuid.uuid4())

    res = _run_only_stage_ingest(orch, job_id=job_id, source_video_paths=[src0, src1])

    assert len(res["conformed_sources"]) == 2
    assert "source_0" in res["conformed_sources"]
    assert "source_1" in res["conformed_sources"]

    for sid in ["source_0", "source_1"]:
        cpath = res["conformed_sources"][sid]
        assert cpath.exists()
        meta = ffmpeg_utils.probe_video(cpath)
        v = next(s for s in meta["streams"] if s["codec_type"] == "video")
        a = next(s for s in meta["streams"] if s["codec_type"] == "audio")
        assert int(v["width"]) == 1080
        assert int(v["height"]) == 1920
        assert int(a["sample_rate"]) == 48000


def test_orchestrator_stage_ingest_5_files_heterogeneous(tmp_path: Path):
    """PipelineOrchestrator _stage_ingest with 5 heterogeneous files (resolutions, framerates, audio presence)."""
    orch = PipelineOrchestrator(MockProvider())
    orch.config.jobs_dir_raw = str(tmp_path / "jobs")

    specs = [
        {"w": 640, "h": 360, "fps": 24, "audio": True},
        {"w": 1080, "h": 1920, "fps": 30, "audio": True},
        {"w": 1280, "h": 720, "fps": 25, "audio": True},
        {"w": 720, "h": 720, "fps": 30, "audio": False},  # Square, no audio
        {"w": 1920, "h": 1080, "fps": 60, "audio": True},
    ]

    files = []
    for idx, s in enumerate(specs):
        p = tmp_path / "raw_5" / f"source_clip_{idx}.mp4"
        create_synthetic_test_video(
            p, width=s["w"], height=s["h"], fps=s["fps"], duration=1.0, has_audio=s["audio"]
        )
        files.append(p)

    job_id = str(uuid.uuid4())
    res = _run_only_stage_ingest(orch, job_id=job_id, source_video_paths=files)

    assert len(res["conformed_sources"]) == 5
    for idx in range(5):
        sid = f"source_{idx}"
        assert sid in res["conformed_sources"]
        cpath = res["conformed_sources"][sid]
        assert cpath.exists()
        meta = ffmpeg_utils.probe_video(cpath)
        v = next(s for s in meta["streams"] if s["codec_type"] == "video")
        a = next(s for s in meta["streams"] if s["codec_type"] == "audio")
        assert int(v["width"]) == 1080
        assert int(v["height"]) == 1920
        assert int(a["sample_rate"]) == 48000
        assert int(a["channels"]) == 2


def test_orchestrator_stage_ingest_empty_and_invalid(tmp_path: Path):
    """PipelineOrchestrator rejects empty lists and invalid video paths."""
    orch = PipelineOrchestrator(MockProvider())
    orch.config.jobs_dir_raw = str(tmp_path / "jobs")

    # Empty list
    with pytest.raises(ValueError, match="source_video_paths cannot be empty"):
        orch.run_job(job_id=str(uuid.uuid4()), source_video_paths=[])

    # Nonexistent file
    with pytest.raises((ValueError, RuntimeError)):
        orch.run_job(
            job_id=str(uuid.uuid4()),
            source_video_paths=[tmp_path / "nonexistent.mp4"],
        )

    # 0-byte file
    zero_file = tmp_path / "zero.mp4"
    zero_file.touch()
    with pytest.raises((ValueError, RuntimeError)):
        orch.run_job(
            job_id=str(uuid.uuid4()),
            source_video_paths=[zero_file],
        )


def test_orchestrator_stage_ingest_basename_collision_exposure(tmp_path: Path):
    """
    Adversarial test: 2 files with the SAME filename in different directories
    demonstrates the filename collision behavior in file_utils.copy_source_video.
    """
    orch = PipelineOrchestrator(MockProvider())
    orch.config.jobs_dir_raw = str(tmp_path / "jobs")

    dir_a = tmp_path / "cam_a"
    dir_b = tmp_path / "cam_b"
    f_a = dir_a / "interview.mp4"
    f_b = dir_b / "interview.mp4"

    # f_a is 1.0s, f_b is 3.0s
    create_synthetic_test_video(f_a, duration=1.0)
    create_synthetic_test_video(f_b, duration=3.0)

    job_id = str(uuid.uuid4())
    res = _run_only_stage_ingest(orch, job_id=job_id, source_video_paths=[f_a, f_b])

    # Probe both conformed outputs
    meta0 = ffmpeg_utils.probe_video(res["conformed_sources"]["source_0"])
    meta1 = ffmpeg_utils.probe_video(res["conformed_sources"]["source_1"])
    dur0 = float(meta0["format"]["duration"])
    dur1 = float(meta1["format"]["duration"])

    # Verify that collision was prevented: source_map paths are distinct and durations differ
    assert "source_0" in res["conformed_sources"]
    assert "source_1" in res["conformed_sources"]
    assert res["source_map"]["source_0"] != res["source_map"]["source_1"]
    assert res["source_map"]["source_0"].name == "source_0_interview.mp4"
    assert res["source_map"]["source_1"].name == "source_1_interview.mp4"
    assert dur0 == pytest.approx(1.0, abs=0.2), f"Expected source_0 to be ~1.0s, got {dur0}"
    assert dur1 == pytest.approx(3.0, abs=0.2), f"Expected source_1 to be ~3.0s, got {dur1}"
    assert abs(dur0 - dur1) > 1.5, "Conformed durations must differ (no overwrite)"
