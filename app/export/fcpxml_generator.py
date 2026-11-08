"""
app/export/fcpxml_generator.py — Generates DaVinci Resolve & FCP-compatible FCPXML timelines.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET
from xml.dom import minidom

from app.schemas.editing_plan import EditingPlan, AspectRatio
from app.schemas.video_meta import VideoMetadata
from app.utils.logger import get_logger

logger = get_logger(__name__)


def generate_fcpxml(
    plan: EditingPlan,
    source_video_path: Path,
    metadata: VideoMetadata,
    output_path: Path,
    project_name: str = "AI Video Studio Timeline",
) -> Path:
    """
    Export an FCPXML (v1.9) timeline file ready for DaVinci Resolve and Final Cut Pro.

    Embeds:
    - Source asset media reference
    - Cut sequence with start/end frame timings
    - Video scale punch-ins (115% hook, alternating zooms)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolution determination based on plan aspect ratio
    if plan.aspect_ratio == AspectRatio.VERTICAL_9_16:
        width, height = 1080, 1920
    elif plan.aspect_ratio == AspectRatio.SQUARE_1_1:
        width, height = 1080, 1080
    else:
        width, height = 1920, 1080

    fps = metadata.frame_rate or 30.0
    frame_dur = f"1/{int(round(fps))}s"

    # Root
    fcpxml = ET.Element("fcpxml", version="1.9")

    # Resources
    resources = ET.SubElement(fcpxml, "resources")
    ET.SubElement(
        resources,
        "format",
        id="r1",
        name=f"Format_{width}x{height}",
        frameDuration=frame_dur,
        width=str(width),
        height=str(height),
    )

    source_uri = source_video_path.resolve().as_uri()
    ET.SubElement(
        resources,
        "asset",
        id="r2",
        name=source_video_path.name,
        src=source_uri,
        start="0s",
        duration=f"{metadata.duration_seconds:.3f}s",
        hasVideo="1",
        hasAudio="1" if metadata.has_audio else "0",
        format="r1",
    )

    # Library / Event / Project / Sequence / Spine
    library = ET.SubElement(fcpxml, "library")
    event = ET.SubElement(library, "event", name="AI Studio Sequences")
    project = ET.SubElement(event, "project", name=project_name)

    total_duration = sum(c.duration for c in plan.clips)
    sequence = ET.SubElement(
        project,
        "sequence",
        format="r1",
        duration=f"{total_duration:.3f}s",
        tcStart="0s",
        tcFormat="NDF",
    )
    spine = ET.SubElement(sequence, "spine")

    # Populate clips in order
    sorted_clips = sorted(plan.clips, key=lambda c: c.order)
    current_offset = 0.0

    for clip in sorted_clips:
        clip_dur = clip.duration
        clip_elem = ET.SubElement(
            spine,
            "clip",
            name=f"Clip_{clip.clip_id}",
            offset=f"{current_offset:.3f}s",
            duration=f"{clip_dur:.3f}s",
            start=f"{clip.source_start:.3f}s",
            ref="r2",
        )

        # Scale transform (zoom / punch-in)
        scale_factor = 1.0
        if clip.zoom.enabled:
            scale_factor = clip.zoom.factor
        elif clip.order == 1:
            scale_factor = 1.15  # Hook punch-in default

        if scale_factor != 1.0:
            ET.SubElement(
                clip_elem,
                "adjust-transform",
                scale=f"{scale_factor:.3f} {scale_factor:.3f}",
                position="0 0",
            )

        current_offset += clip_dur

    # Pretty format XML
    raw_xml = ET.tostring(fcpxml, encoding="utf-8")
    parsed = minidom.parseString(raw_xml)
    pretty_xml = parsed.toprettyxml(indent="    ", encoding="utf-8").decode("utf-8")

    output_path.write_text(pretty_xml, encoding="utf-8")
    logger.info("Generated FCPXML timeline: %s (%d clips)", output_path, len(plan.clips))
    return output_path
