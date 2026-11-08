"""
tests/test_pipeline.py — Unit tests for pipeline utilities and stage safety.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from app.utils.file_utils import create_job_dirs, safe_job_path


def test_safe_job_path_allows_valid_subpaths(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    job_id = "job-123"
    p = safe_job_path(jobs_dir, job_id, "exports", "final.mp4")
    assert str(p).startswith(str(jobs_dir.resolve()))


def test_safe_job_path_blocks_directory_traversal(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    job_id = "job-123"
    with pytest.raises(ValueError, match="Path traversal detected"):
        safe_job_path(jobs_dir, job_id, "..", "..", "etc", "passwd")


def test_create_job_dirs_creates_all_folders(tmp_path: Path):
    jobs_dir = tmp_path / "jobs"
    job_id = "job-456"
    dirs = create_job_dirs(jobs_dir, job_id)

    expected = ["source", "analysis", "transcripts", "plans", "intermediate", "previews", "exports"]
    for name in expected:
        assert name in dirs
        assert dirs[name].is_dir()
