"""
app/utils/file_utils.py — Path validation and safe file operations.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Optional

from app.config import get_config
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Allowed video suffixes (lowercase)
_ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov"}


def validate_video_path(path: str) -> Path:
    """
    Validate that a video file exists, has an allowed extension, and is within
    the configured size limit.

    Parameters
    ----------
    path:
        Absolute or relative path to the video file.

    Returns
    -------
    Path
        Resolved absolute path.

    Raises
    ------
    ValueError
        If the file does not exist, has an unsupported extension, or exceeds
        the configured maximum file size.
    """
    p = Path(path).resolve()

    if not p.exists():
        raise ValueError(f"Video file does not exist: {p}")

    if not p.is_file():
        raise ValueError(f"Path is not a file: {p}")

    if p.suffix.lower() not in _ALLOWED_VIDEO_SUFFIXES:
        raise ValueError(
            f"Unsupported video format '{p.suffix}'. "
            f"Allowed formats: {', '.join(_ALLOWED_VIDEO_SUFFIXES)}"
        )

    cfg = get_config()
    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > cfg.max_video_size_mb:
        raise ValueError(
            f"File size {size_mb:.1f} MB exceeds maximum allowed "
            f"{cfg.max_video_size_mb} MB."
        )

    logger.debug("Validated video path: %s (%.1f MB)", p, size_mb)
    return p


def safe_job_path(jobs_dir: Path, job_id: str, *parts: str) -> Path:
    """
    Construct a path inside the jobs directory and validate that it stays
    within the jobs directory (prevents path traversal attacks).

    Parameters
    ----------
    jobs_dir:
        Absolute path to the top-level jobs directory.
    job_id:
        Job UUID string.
    *parts:
        Additional path components appended after the job directory.

    Returns
    -------
    Path
        Resolved absolute path guaranteed to be under ``jobs_dir``.

    Raises
    ------
    ValueError
        If the resolved path escapes the jobs directory.
    """
    resolved_jobs_dir = jobs_dir.resolve()
    candidate = (resolved_jobs_dir / job_id / Path(*parts) if parts else resolved_jobs_dir / job_id).resolve()

    try:
        candidate.relative_to(resolved_jobs_dir)
    except ValueError:
        raise ValueError(
            f"Path traversal detected: '{candidate}' is not inside '{resolved_jobs_dir}'"
        )

    return candidate


def create_job_dirs(jobs_dir: Path, job_id: str) -> Dict[str, Path]:
    """
    Create all required subdirectories for a new job.

    Parameters
    ----------
    jobs_dir:
        Absolute path to the top-level jobs directory.
    job_id:
        Job UUID string.

    Returns
    -------
    dict[str, Path]
        Mapping of directory name → absolute Path for each subdirectory.
    """
    subdirs = [
        "source",
        "analysis",
        "transcripts",
        "plans",
        "intermediate",
        "previews",
        "exports",
    ]

    result: Dict[str, Path] = {}
    for name in subdirs:
        p = safe_job_path(jobs_dir, job_id, name)
        p.mkdir(parents=True, exist_ok=True)
        result[name] = p

    # Also store the job root itself
    result["root"] = safe_job_path(jobs_dir, job_id)

    logger.debug("Created job directories for job_id=%s: %s", job_id, list(result.keys()))
    return result


def copy_source_video(source_path: Path, job_source_dir: Path, dest_name: Optional[str] = None) -> Path:
    """
    Copy the original video into the job's source directory, preserving the
    original filename unless dest_name is provided.

    Parameters
    ----------
    source_path:
        Absolute path to the original video.
    job_source_dir:
        Path to the job's ``source/`` subdirectory.
    dest_name:
        Optional custom filename for the copied destination file.

    Returns
    -------
    Path
        Destination path of the copied file.
    """
    target_name = dest_name if dest_name is not None else source_path.name
    dest = job_source_dir / target_name
    shutil.copy2(source_path, dest)
    logger.info("Copied source video to %s", dest)
    return dest
