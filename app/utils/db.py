"""
app/utils/db.py — SQLite job tracking database.

Schema: jobs table with per-job status, stage, and metadata.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Generator, List, Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)

_CREATE_JOBS_TABLE = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id        TEXT PRIMARY KEY,
    status        TEXT NOT NULL DEFAULT 'PENDING',
    profile       TEXT NOT NULL,
    source_path   TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    current_stage TEXT,
    error         TEXT,
    metadata_json TEXT
);
"""


@contextmanager
def _get_conn(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """Context manager that opens a SQLite connection and auto-commits/rolls back."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(db_path: Path) -> None:
    """
    Initialise the database, creating tables if they do not exist.

    Parameters
    ----------
    db_path:
        Absolute path to the SQLite database file.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _get_conn(db_path) as conn:
        conn.execute(_CREATE_JOBS_TABLE)
    logger.debug("Database initialised at %s", db_path)


def create_job(
    db_path: Path,
    job_id: str,
    profile: str,
    source_path: str,
) -> None:
    """
    Insert a new job record with PENDING status.

    Parameters
    ----------
    db_path:
        Path to the SQLite database.
    job_id:
        Unique job UUID string.
    profile:
        Editing profile name.
    source_path:
        Absolute path to the source video file.
    """
    now = _now_iso()
    with _get_conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO jobs (job_id, status, profile, source_path, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (job_id, "PENDING", profile, source_path, now, now),
        )
    logger.info("Created job %s (profile=%s)", job_id, profile)


def update_job_status(
    db_path: Path,
    job_id: str,
    status: str,
    current_stage: Optional[str] = None,
    error: Optional[str] = None,
    metadata: Optional[dict] = None,
    metadata_json: Optional[str] = None,
) -> None:
    """
    Update the status and optional fields of an existing job.
    """
    final_meta_json = metadata_json
    if final_meta_json is None and metadata is not None:
        final_meta_json = json.dumps(metadata)

    with _get_conn(db_path) as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status = ?,
                updated_at = ?,
                current_stage = COALESCE(?, current_stage),
                error = COALESCE(?, error),
                metadata_json = COALESCE(?, metadata_json)
            WHERE job_id = ?
            """,
            (status, _now_iso(), current_stage, error, final_meta_json, job_id),
        )
    logger.debug("Updated job %s → status=%s stage=%s", job_id, status, current_stage)


def get_job(db_path: Path, job_id: str) -> Optional[Dict]:
    """
    Fetch a single job record by ID.

    Parameters
    ----------
    db_path:
        Path to the SQLite database.
    job_id:
        Job UUID string.

    Returns
    -------
    dict | None
        Job record as a dict, or ``None`` if not found.
    """
    with _get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
        ).fetchone()

    if row is None:
        return None

    record = dict(row)
    if record.get("metadata_json"):
        try:
            record["metadata"] = json.loads(record["metadata_json"])
        except json.JSONDecodeError:
            record["metadata"] = {}
    else:
        record["metadata"] = {}
    return record


def list_jobs(db_path: Path, limit: int = 50) -> List[Dict]:
    """
    Return the most recent jobs, ordered by creation time descending.

    Parameters
    ----------
    db_path:
        Path to the SQLite database.
    limit:
        Maximum number of jobs to return.

    Returns
    -------
    list[dict]
        List of job records.
    """
    with _get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()

    result = []
    for row in rows:
        record = dict(row)
        if record.get("metadata_json"):
            try:
                record["metadata"] = json.loads(record["metadata_json"])
            except json.JSONDecodeError:
                record["metadata"] = {}
        else:
            record["metadata"] = {}
        result.append(record)
    return result
