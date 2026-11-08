"""
app/schemas/job.py — Job state Pydantic models.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class JobStatus(str, Enum):
    """All possible states a job can be in."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    TRANSCRIBING = "TRANSCRIBING"
    ANALYZING = "ANALYZING"
    PLANNING = "PLANNING"
    EDITING = "EDITING"
    RENDERING = "RENDERING"
    DONE = "DONE"
    FAILED = "FAILED"


class StageProgress(BaseModel):
    """Progress record for a single pipeline stage."""

    stage_name: str
    status: Literal["pending", "running", "done", "failed", "skipped"]
    message: str = ""
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class Job(BaseModel):
    """Full job record combining database fields with structured stage progress."""

    job_id: str
    status: JobStatus
    profile: str
    source_path: str = ""
    source_paths: List[str] = Field(default_factory=list)
    created_at: str
    updated_at: str
    stages: List[StageProgress] = Field(default_factory=list)
    error: Optional[str] = None
    metadata: Optional[dict] = None

    @model_validator(mode="before")
    @classmethod
    def _sync_sources_before(cls, data: Any) -> Any:
        if isinstance(data, dict):
            paths = data.get("source_paths")
            single = data.get("source_path")
            if paths and not single:
                data["source_path"] = str(paths[0])
            elif single and not paths:
                data["source_paths"] = [str(single)]
            elif not paths and not single:
                data.setdefault("source_path", "")
                data.setdefault("source_paths", [])
        return data

    @model_validator(mode="after")
    def _sync_sources_after(self) -> "Job":
        if self.source_paths and not self.source_path:
            self.source_path = self.source_paths[0]
        elif self.source_path and not self.source_paths:
            self.source_paths = [self.source_path]
        return self

    @property
    def primary_source_path(self) -> str:
        """Return the primary source video path."""
        if self.source_paths:
            return self.source_paths[0]
        return self.source_path
