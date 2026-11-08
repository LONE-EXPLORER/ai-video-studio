"""
app/schemas/director_plan.py — Schema for Google AI Studio Director Plan.

Defines the structure for advanced visual asset generation instructions:
- Geopolitical animated maps
- Financial / data animated charts
- Free stock footage / B-roll search queries
- Sound effects & punch-in framing cues
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class AssetType(str, Enum):
    MAP = "map"
    CHART = "chart"
    STOCK_VIDEO = "stock_video"
    KINETIC_TEXT = "kinetic_text"


class MapGenerationCue(BaseModel):
    cue_id: str
    timeline_start: float
    duration: float = Field(default=3.0, ge=1.0, le=8.0)
    region_name: str = Field(description="e.g. Middle East, Strait of Hormuz, Persian Gulf, India")
    highlight_countries: List[str] = Field(default_factory=list, description="e.g. ['Iran', 'Israel', 'USA']")
    route_arrows: List[str] = Field(default_factory=list, description="e.g. ['Iran -> Strait of Hormuz']")
    overlay_title: str = Field(description="Headline to display on map, e.g. 'THE 5-DAY BLOCKADE'")
    motion_type: str = Field(default="zoom_in", description="zoom_in, pan_left, radar_sweep")


class ChartGenerationCue(BaseModel):
    cue_id: str
    timeline_start: float
    duration: float = Field(default=3.0, ge=1.0, le=8.0)
    chart_type: str = Field(default="line", description="line, bar, metric_counter")
    metric_title: str = Field(description="e.g. 'Crude Oil Price ($/barrel)'")
    data_labels: List[str] = Field(description="e.g. ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5']")
    data_values: List[float] = Field(description="e.g. [72.0, 78.5, 92.0, 115.0, 140.0]")
    callout_text: str = Field(default="", description="e.g. '+94% Surge'")


class StockFootageCue(BaseModel):
    cue_id: str
    timeline_start: float
    duration: float = Field(default=2.5, ge=1.0, le=6.0)
    search_keywords: List[str] = Field(description="e.g. ['oil tanker', 'warship ocean', 'radar screen']")
    visual_description: str = Field(description="Detailed visual prompt describing the scene")
    overlay_mode: str = Field(default="cutaway", description="cutaway or pip")


class GeminiDirectorPlan(BaseModel):
    """Master editorial direction emitted by Google AI Studio."""
    project_title: str
    hook_concept: str = Field(description="First 3 seconds retention hook strategy")
    target_pacing_style: str = Field(default="relentless_shorts", description="Pacing profile")
    
    # Visual cues to be generated locally
    map_cues: List[MapGenerationCue] = Field(default_factory=list)
    chart_cues: List[ChartGenerationCue] = Field(default_factory=list)
    stock_cues: List[StockFootageCue] = Field(default_factory=list)
    
    editorial_notes: str = Field(default="")
