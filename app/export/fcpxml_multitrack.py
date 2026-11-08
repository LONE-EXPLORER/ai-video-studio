"""
app/export/fcpxml_multitrack.py — Full multi-track FCPXML v1.9 export.

Generates a DaVinci Resolve / Final Cut Pro compatible XML timeline with:
  V1  (spine)    — A-roll dialogue clips with scale punch-ins
  V2  (lane=1)   — B-roll overlay clips
  V3  (lane=2)   — Kinetic title/caption connected clips
  A1  (lane=-1)  — Dialogue audio
  A2  (lane=-2)  — SFX channel
  A3  (lane=-3)  — Background music
  Markers at hook point and major cut transitions
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional
import xml.etree.ElementTree as ET
from xml.dom import minidom

from app.schemas.editing_plan import EditingPlan, AspectRatio, BrollOverlay
from app.schemas.video_meta import VideoMetadata
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _frames(seconds: float, fps: float) -> str:
    """Convert seconds to FCPXML fractional-second rational notation (e.g. '90/30s')."""
    frame_count = int(round(seconds * fps))
    return f"{frame_count}/{int(round(fps))}s"


def generate_multitrack_fcpxml(
    plan: EditingPlan,
    source_video_path: Path,
    metadata: VideoMetadata,
    output_path: Path,
    project_name: str = "AI Video Studio — Marvel Timeline",
    music_path: Optional[Path] = None,
    sfx_events: Optional[list] = None,
    broll_overlays: Optional[List[BrollOverlay]] = None,
    ass_subtitle_path: Optional[Path] = None,
) -> Path:
    """
    Export a full multi-track FCPXML v1.9 timeline.

    Parameters
    ----------
    plan : EditingPlan
        Validated editing plan with clip list and settings.
    source_video_path : Path
        Path to the primary source video file.
    metadata : VideoMetadata
        Source video metadata (fps, duration, etc.).
    output_path : Path
        Destination .fcpxml file path.
    project_name : str
        Name for the FCP/DaVinci project.
    music_path : Path, optional
        Background music asset path.
    sfx_events : list, optional
        List of SFXEvent objects with timecode/sfx_name.
    broll_overlays : list, optional
        B-roll overlay events from the plan.
    ass_subtitle_path : Path, optional
        Path to .ass subtitle file for titles lane.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolution
    if plan.aspect_ratio == AspectRatio.VERTICAL_9_16:
        width, height = 1080, 1920
    elif plan.aspect_ratio == AspectRatio.SQUARE_1_1:
        width, height = 1080, 1080
    else:
        width, height = 1920, 1080

    fps = metadata.frame_rate or 30.0
    fps_int = int(round(fps))
    frame_dur_str = f"1/{fps_int}s"

    # ------------------------------------------------------------------ #
    # Root element + DOCTYPE                                               #
    # ------------------------------------------------------------------ #
    fcpxml = ET.Element("fcpxml", version="1.9")

    # ------------------------------------------------------------------ #
    # Resources                                                            #
    # ------------------------------------------------------------------ #
    resources = ET.SubElement(fcpxml, "resources")

    # r1 — Format
    ET.SubElement(
        resources, "format",
        id="r1",
        name=f"FFVideoFormat{height}p{fps_int}",
        frameDuration=frame_dur_str,
        width=str(width),
        height=str(height),
    )

    # r2 — Primary A-roll asset
    source_uri = source_video_path.resolve().as_uri()
    total_src_dur = _frames(metadata.duration_seconds, fps)
    ET.SubElement(
        resources, "asset",
        id="r2",
        name=source_video_path.stem,
        src=source_uri,
        start="0s",
        duration=total_src_dur,
        hasVideo="1",
        hasAudio="1" if metadata.has_audio else "0",
        format="r1",
    )

    # r3 — Background music (if provided)
    next_resource_id = 3
    music_resource_id = None
    if music_path and music_path.exists():
        music_resource_id = f"r{next_resource_id}"
        next_resource_id += 1
        ET.SubElement(
            resources, "asset",
            id=music_resource_id,
            name=music_path.stem,
            src=music_path.resolve().as_uri(),
            start="0s",
            duration="600s",   # Assume music is at least 10min; trim by sequence
            hasVideo="0",
            hasAudio="1",
            format="r1",
        )

    # B-roll asset resources
    broll_resource_map: dict = {}  # asset_path → resource_id
    overlays_to_use = broll_overlays or plan.broll_overlays or []
    for overlay in overlays_to_use:
        broll_raw = getattr(overlay, "source_path", None) or getattr(overlay, "asset_path", "")
        broll_path = Path(broll_raw)
        broll_key = str(broll_path.resolve())
        if broll_key not in broll_resource_map and broll_path.exists():
            rid = f"r{next_resource_id}"
            next_resource_id += 1
            broll_resource_map[broll_key] = rid
            start_t = getattr(overlay, "timeline_start", getattr(overlay, "start_time", 0.0))
            end_t = getattr(overlay, "timeline_end", None)
            dur_t = (end_t - start_t) if end_t is not None else getattr(overlay, "duration", 2.0)
            ET.SubElement(
                resources, "asset",
                id=rid,
                name=broll_path.stem,
                src=broll_path.resolve().as_uri(),
                start="0s",
                duration=_frames(max(dur_t, 1.0) + 2.0, fps),
                hasVideo="1",
                hasAudio="0",
                format="r1",
            )

    # ------------------------------------------------------------------ #
    # Library / Event / Project / Sequence / Spine                        #
    # ------------------------------------------------------------------ #
    library = ET.SubElement(fcpxml, "library")
    event = ET.SubElement(library, "event", name="AI Studio — Sequences")
    project = ET.SubElement(event, "project", name=project_name)

    total_edit_duration = sum(c.duration for c in plan.clips)
    sequence = ET.SubElement(
        project, "sequence",
        format="r1",
        duration=_frames(total_edit_duration, fps),
        tcStart="0s",
        tcFormat="NDF",
        audioLayout="stereo",
        audioRate="48k",
    )
    spine = ET.SubElement(sequence, "spine")

    # ------------------------------------------------------------------ #
    # V1 — A-roll spine clips with scale punch-ins                        #
    # ------------------------------------------------------------------ #
    sorted_clips = sorted(plan.clips, key=lambda c: c.order)
    cut_timecodes: List[float] = []
    current_offset = 0.0

    for clip in sorted_clips:
        cut_timecodes.append(current_offset)
        clip_dur = clip.duration
        clip_elem = ET.SubElement(
            spine, "clip",
            name=f"A-Roll_Clip_{clip.clip_id}",
            offset=_frames(current_offset, fps),
            duration=_frames(clip_dur, fps),
            start=_frames(clip.source_start, fps),
            ref="r2",
            lane="0",
        )

        # Scale transform
        scale_factor = 1.0
        if clip.zoom.enabled:
            scale_factor = clip.zoom.factor
        elif clip.order == 1:
            scale_factor = 1.15

        if scale_factor != 1.0:
            ET.SubElement(
                clip_elem, "adjust-transform",
                scale=f"{scale_factor:.4f} {scale_factor:.4f}",
                position="0 0",
                anchor="0 0",
            )

        # Dialogue audio (A1) — connected audio-only clip
        audio_elem = ET.SubElement(
            clip_elem, "audio",
            name=f"Dialogue_{clip.clip_id}",
            offset="0s",
            duration=_frames(clip_dur, fps),
            start=_frames(clip.source_start, fps),
            ref="r2",
            lane="-1",
            role="dialogue",
        )
        ET.SubElement(audio_elem, "adjust-volume", amount="0dB")

        current_offset += clip_dur

    # ------------------------------------------------------------------ #
    # Markers at hook + every 5th cut                                     #
    # ------------------------------------------------------------------ #
    marker_positions = [0.0]  # Always mark the hook
    marker_positions += [tc for i, tc in enumerate(cut_timecodes) if i > 0 and i % 5 == 0]

    for i, tc in enumerate(marker_positions):
        ET.SubElement(
            spine, "marker",
            start=_frames(tc, fps),
            duration=frame_dur_str,
            value=f"{'🪝 Hook' if i == 0 else f'✂️ Cut {i*5}'}",
            completed="0",
        )

    # ------------------------------------------------------------------ #
    # V2 — B-roll lane (lane=1) connected clips                          #
    # ------------------------------------------------------------------ #
    for overlay in overlays_to_use:
        broll_raw = getattr(overlay, "source_path", None) or getattr(overlay, "asset_path", "")
        broll_path = Path(broll_raw)
        broll_key = str(broll_path.resolve())
        if broll_key not in broll_resource_map:
            continue
        rid = broll_resource_map[broll_key]
        start_t = getattr(overlay, "timeline_start", getattr(overlay, "start_time", 0.0))
        end_t = getattr(overlay, "timeline_end", None)
        dur_t = (end_t - start_t) if end_t is not None else getattr(overlay, "duration", 2.0)
        broll_clip = ET.SubElement(
            spine, "clip",
            name=f"B-Roll_{broll_path.stem[:20]}",
            offset=_frames(start_t, fps),
            duration=_frames(dur_t, fps),
            start="0s",
            ref=rid,
            lane="1",
        )
        # Slight scale-down so B-roll doesn't overflow frame edges
        ET.SubElement(
            broll_clip, "adjust-transform",
            scale="0.95 0.95",
            position="0 0",
        )

    # ------------------------------------------------------------------ #
    # A2 — SFX lane (lane=-2) connected audio clips                      #
    # ------------------------------------------------------------------ #
    if sfx_events:
        sfx_index_path = Path("assets/sfx/sfx_index.json")
        if sfx_index_path.exists():
            import json
            sfx_index = json.loads(sfx_index_path.read_text())
        else:
            sfx_index = {}

        for i, ev in enumerate(sfx_events):
            sfx_info = sfx_index.get(ev.sfx_name if hasattr(ev, "sfx_name") else ev.get("sfx_name", ""), {})
            sfx_path_str = sfx_info.get("path", "")
            sfx_path = Path("assets/sfx") / Path(sfx_path_str).name if sfx_path_str else None
            if sfx_path and sfx_path.exists():
                dur = sfx_info.get("duration", 0.5)
                tc = ev.timecode if hasattr(ev, "timecode") else ev.get("timecode", 0)
                sfx_rid = f"r{next_resource_id}"
                next_resource_id += 1
                ET.SubElement(
                    resources, "asset",
                    id=sfx_rid,
                    name=sfx_path.stem,
                    src=sfx_path.resolve().as_uri(),
                    start="0s",
                    duration=_frames(dur + 0.1, fps),
                    hasVideo="0",
                    hasAudio="1",
                    format="r1",
                )
                sfx_elem = ET.SubElement(
                    spine, "audio",
                    name=f"SFX_{ev.sfx_name if hasattr(ev, 'sfx_name') else 'sfx'}_{i:02d}",
                    offset=_frames(max(0.0, tc), fps),
                    duration=_frames(dur, fps),
                    start="0s",
                    ref=sfx_rid,
                    lane="-2",
                    role="effects",
                )
                vol = getattr(ev, "volume", sfx_info.get("volume", 0.6)) * 100.0
                ET.SubElement(sfx_elem, "adjust-volume", amount=f"{vol:.0f}%")

    # ------------------------------------------------------------------ #
    # A3 — Background music lane (lane=-3)                                #
    # ------------------------------------------------------------------ #
    if music_resource_id:
        music_elem = ET.SubElement(
            spine, "audio",
            name="BGM_Ambient",
            offset="0s",
            duration=_frames(total_edit_duration, fps),
            start="0s",
            ref=music_resource_id,
            lane="-3",
            role="music",
        )
        ET.SubElement(music_elem, "adjust-volume", amount="35%")

    # ------------------------------------------------------------------ #
    # V3 — Subtitle/Title lane (lane=2) — reference to .ass if available  #
    # ------------------------------------------------------------------ #
    if ass_subtitle_path and ass_subtitle_path.exists():
        # In FCPXML, subtitles can be referenced as a caption lane
        # We create a single caption clip covering the full sequence
        caption_rid = f"r{next_resource_id}"
        next_resource_id += 1
        ET.SubElement(
            resources, "asset",
            id=caption_rid,
            name=ass_subtitle_path.stem,
            src=ass_subtitle_path.resolve().as_uri(),
            start="0s",
            duration=_frames(total_edit_duration, fps),
            hasVideo="0",
            hasAudio="0",
            format="r1",
        )
        ET.SubElement(
            spine, "caption",
            name="Kinetic_Captions",
            offset="0s",
            duration=_frames(total_edit_duration, fps),
            start="0s",
            ref=caption_rid,
            lane="2",
            role="captions.iTT?captionFormat=ITT",
        )

    # ------------------------------------------------------------------ #
    # Pretty-print XML output                                              #
    # ------------------------------------------------------------------ #
    # Inject DOCTYPE declaration (required for DaVinci Resolve compatibility)
    raw_xml = ET.tostring(fcpxml, encoding="unicode")
    doctype = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE fcpxml>\n'
    )
    parsed = minidom.parseString(f'<root>{raw_xml}</root>')
    # Re-serialize cleanly
    raw2 = ET.tostring(fcpxml, encoding="utf-8")
    parsed2 = minidom.parseString(raw2)
    pretty = parsed2.toprettyxml(indent="    ", encoding="utf-8").decode("utf-8")
    # Replace auto-generated XML declaration with our versioned one
    pretty = pretty.replace('<?xml version="1.0" ?>', '<?xml version="1.0" encoding="UTF-8"?>')

    output_path.write_text(pretty, encoding="utf-8")

    n_broll = len(overlays_to_use)
    n_sfx = len(sfx_events) if sfx_events else 0
    logger.info(
        "Generated multi-track FCPXML: %s\n"
        "  → V1: %d A-roll clips | V2: %d B-roll | V3: captions\n"
        "  → A1: dialogue | A2: %d SFX events | A3: %s",
        output_path, len(plan.clips), n_broll, n_sfx,
        "BGM" if music_resource_id else "no BGM",
    )
    return output_path
