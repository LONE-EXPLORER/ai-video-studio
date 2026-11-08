"""
app/rendering/preview_gen.py — Lower-bitrate preview video generation.
"""

from __future__ import annotations

from pathlib import Path

from app.utils.ffmpeg_utils import generate_preview as _generate_preview
from app.utils.logger import get_logger

logger = get_logger(__name__)


def generate_preview(input_path: Path, output_path: Path) -> Path:
    """
    Generate a web-streamable preview video at reduced bitrate.

    Uses:
    - Maximum width of 854 px (height scaled proportionally)
    - CRF 28 (lower quality, faster encode)
    - 1500k video bitrate cap
    - ``fast`` encoder preset
    - ``+faststart`` moov atom placement for browser streaming

    Parameters
    ----------
    input_path:
        Source video file (typically the final rendered export).
    output_path:
        Destination preview MP4 path.

    Returns
    -------
    Path
        Path to the generated preview file.
    """
    result = _generate_preview(input_path, output_path, max_width=854)
    logger.info("Preview generated: %s", output_path)
    return result
