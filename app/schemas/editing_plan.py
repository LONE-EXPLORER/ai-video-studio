"""
app/schemas/editing_plan.py — EditingPlan, Clip, CaptionSettings Pydantic models.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


class AspectRatio(str, Enum):
    """Supported output aspect ratios."""

    VERTICAL_9_16 = "9:16"
    LANDSCAPE_16_9 = "16:9"
    SQUARE_1_1 = "1:1"


class TransitionType(str, Enum):
    """Clip transition types."""

    CUT = "cut"
    FADE = "fade"
    DISSOLVE = "dissolve"


class ZoomEffect(BaseModel):
    """Zoom animation applied to a clip."""

    enabled: bool = False
    factor: float = Field(default=1.05, ge=1.0, le=1.3)
    start_second: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def _support_scale_alias(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "scale" in data and "factor" not in data:
                data["factor"] = data.pop("scale")
        return data

    @property
    def scale(self) -> float:
        """Alias for factor."""
        return self.factor


ZoomSettings = ZoomEffect  # Compatibility alias for ZoomSettings


class CaptionSettings(BaseModel):
    """Settings controlling caption generation and appearance."""

    enabled: bool = True
    style: Literal["dynamic", "clean", "minimal", "highlighted"] = "dynamic"
    font_size: int = Field(default=52, ge=20, le=120)
    position: Literal["bottom", "top", "middle"] = "bottom"
    word_highlight_color: str = "#FFD700"
    text_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    margin_bottom: int = 80


class AudioSettings(BaseModel):
    """Settings controlling audio processing."""

    normalize: bool = True
    fade_in: float = 0.3
    fade_out: float = 0.5
    background_music_path: Optional[str] = None
    music_volume: float = Field(default=0.15, ge=0.0, le=1.0)
    duck_music: bool = True


class BrollOverlay(BaseModel):
    """Secondary B-roll overlay cutaway or picture-in-picture placed over the timeline."""

    broll_id: str
    timeline_start: float = Field(ge=0.0)
    timeline_end: float = Field(gt=0.0)
    source_path: str
    overlay_type: Literal["cutaway", "pip"] = "cutaway"

    @property
    def duration(self) -> float:
        return self.timeline_end - self.timeline_start

    @model_validator(mode="after")
    def _validate_overlay_times(self) -> "BrollOverlay":
        if not math.isfinite(self.timeline_start) or not math.isfinite(self.timeline_end):
            raise ValueError(
                f"BrollOverlay '{self.broll_id}': timeline_start and timeline_end must be finite numbers"
            )
        if self.timeline_end <= self.timeline_start:
            raise ValueError(
                f"BrollOverlay '{self.broll_id}': timeline_end ({self.timeline_end}) "
                f"must be greater than timeline_start ({self.timeline_start})"
            )
        return self


class SoundEffect(BaseModel):
    """Audio SFX cue placed at a specific timeline timestamp."""

    sfx_id: str
    timeline_timestamp: float = Field(ge=0.0)
    name: str  # e.g., "whoosh", "pop", "impact", "riser"
    sfx_path: str
    volume: float = Field(default=0.6, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_sfx_times(self) -> "SoundEffect":
        if not math.isfinite(self.timeline_timestamp) or not math.isfinite(self.volume):
            raise ValueError(
                f"SoundEffect '{self.sfx_id}': timeline_timestamp and volume must be finite numbers"
            )
        return self


class Clip(BaseModel):
    """A single clip within an editing plan, referencing source timestamps."""

    clip_id: str
    source_id: str = "source_0"
    source_path: Optional[str] = None
    source_start: float = Field(ge=0.0)
    source_end: float = Field(gt=0.0)
    order: int
    reason: str
    zoom: ZoomEffect = Field(default_factory=ZoomEffect)
    transition_in: TransitionType = TransitionType.CUT
    transition_out: TransitionType = TransitionType.CUT
    audio_lead: float = Field(default=0.0, ge=0.0, description="J-cut audio lead in seconds")
    audio_tail: float = Field(default=0.0, ge=0.0, description="L-cut audio tail in seconds")

    @property
    def duration(self) -> float:
        """Duration of this clip in seconds."""
        return self.source_end - self.source_start

    @model_validator(mode="after")
    def _validate_times(self) -> "Clip":
        if not math.isfinite(self.source_start) or not math.isfinite(self.source_end):
            raise ValueError(
                f"Clip '{self.clip_id}': source_start and source_end must be finite numbers"
            )
        if not math.isfinite(self.audio_lead) or not math.isfinite(self.audio_tail):
            raise ValueError(
                f"Clip '{self.clip_id}': audio_lead and audio_tail must be finite numbers"
            )
        if self.source_end <= self.source_start:
            raise ValueError(
                f"Clip '{self.clip_id}': source_end ({self.source_end}) must be "
                f"greater than source_start ({self.source_start})"
            )
        return self


class EditingPlan(BaseModel):
    """
    Complete editing plan produced by the editorial agent and executed by the pipeline.
    """

    plan_id: str
    job_id: str
    profile: Literal["dynamic_shorts", "long_form", "cinematic"]
    aspect_ratio: AspectRatio
    target_duration_seconds: float
    clips: List[Clip]
    source_map: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from source_id (e.g. 'source_0') to resolved file path",
    )
    broll_overlays: List[BrollOverlay] = Field(default_factory=list)
    sound_effects: List[SoundEffect] = Field(default_factory=list)
    caption_settings: CaptionSettings = Field(default_factory=CaptionSettings)
    audio_settings: AudioSettings = Field(default_factory=AudioSettings)
    hook_summary: str
    edit_rationale: str

    @property
    def total_duration(self) -> float:
        """Sum of all clip durations."""
        return sum(c.duration for c in self.clips)

    def validate_timestamps(
        self,
        source_durations: Union[float, Dict[str, float]],
    ) -> List[str]:
        """
        Check that all clip timestamps are valid relative to the source video(s).

        Parameters
        ----------
        source_durations:
            Either a single float (legacy duration for all clips/source_0) or
            a dictionary mapping source_id -> duration in seconds.

        Returns
        -------
        list[str]
            List of human-readable error strings. Empty if all timestamps are valid.
        """
        errors: List[str] = []
        is_dict = isinstance(source_durations, dict)

        for clip in self.clips:
            if clip.source_start < 0:
                errors.append(
                    f"Clip {clip.clip_id}: source_start {clip.source_start} < 0"
                )

            if is_dict and clip.source_id not in source_durations:
                errors.append(
                    f"Clip {clip.clip_id}: unknown source_id '{clip.source_id}' "
                    f"not present in source durations"
                )
            else:
                max_dur = (
                    float(source_durations[clip.source_id])
                    if is_dict
                    else float(source_durations)
                )
                if clip.source_end > max_dur:
                    errors.append(
                        f"Clip {clip.clip_id} (source '{clip.source_id}'): "
                        f"source_end {clip.source_end} exceeds source duration {max_dur}"
                    )

            if clip.source_end <= clip.source_start:
                errors.append(
                    f"Clip {clip.clip_id}: source_end <= source_start"
                )
        return errors
