"""
app/analysis/frame_extractor.py — Keyframe extraction and contact sheet generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from app.utils.logger import get_logger

logger = get_logger(__name__)


def extract_keyframes(
    video_path: Path,
    output_dir: Path,
    interval_seconds: float = 5.0,
) -> List[Path]:
    """
    Extract one keyframe every *interval_seconds* from *video_path* using FFmpeg.

    Parameters
    ----------
    video_path:
        Source video file.
    output_dir:
        Directory in which JPEG frames will be saved.
    interval_seconds:
        Time between extracted frames in seconds (default 5).

    Returns
    -------
    list[Path]
        Sorted list of extracted JPEG paths.
    """
    from app.utils.ffmpeg_utils import extract_frames  # noqa: PLC0415

    output_dir.mkdir(parents=True, exist_ok=True)
    fps = 1.0 / max(interval_seconds, 0.1)
    frames = extract_frames(video_path, output_dir, fps=fps)
    logger.info(
        "Extracted %d keyframes at %.1f s intervals from %s",
        len(frames),
        interval_seconds,
        video_path,
    )
    return frames


def generate_contact_sheet(
    frame_paths: List[Path],
    output_path: Path,
    cols: int = 4,
) -> Path:
    """
    Compose a contact sheet image from a list of frame images using Pillow.

    Parameters
    ----------
    frame_paths:
        Ordered list of JPEG frame paths.
    output_path:
        Destination JPEG path for the contact sheet.
    cols:
        Number of columns in the grid (default 4).

    Returns
    -------
    Path
        Path to the saved contact sheet image.
    """
    from PIL import Image  # noqa: PLC0415

    if not frame_paths:
        raise ValueError("frame_paths must not be empty")

    # Determine thumbnail size from the first frame
    thumb_w, thumb_h = 320, 180
    try:
        with Image.open(frame_paths[0]) as img:
            aspect = img.width / img.height
            thumb_h = int(thumb_w / aspect)
    except Exception as exc:
        logger.warning("Could not determine frame dimensions: %s", exc)

    rows = (len(frame_paths) + cols - 1) // cols
    sheet_w = thumb_w * cols
    sheet_h = thumb_h * rows

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(30, 30, 30))

    for idx, frame_path in enumerate(frame_paths):
        col = idx % cols
        row = idx // cols
        try:
            with Image.open(frame_path) as img:
                thumb = img.resize((thumb_w, thumb_h), Image.LANCZOS)
                sheet.paste(thumb, (col * thumb_w, row * thumb_h))
        except Exception as exc:
            logger.warning("Skipping frame %s: %s", frame_path.name, exc)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(str(output_path), "JPEG", quality=85)
    logger.info("Generated contact sheet (%dx%d) → %s", sheet_w, sheet_h, output_path)
    return output_path
