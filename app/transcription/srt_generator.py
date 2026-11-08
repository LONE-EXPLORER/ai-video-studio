"""
app/transcription/srt_generator.py — SRT and ASS subtitle file generation.
"""

from __future__ import annotations

from pathlib import Path

from app.schemas.transcript import Transcript
from app.utils.logger import get_logger

logger = get_logger(__name__)


def generate_srt(transcript: Transcript, output_path: Path) -> Path:
    """
    Write an SRT subtitle file from a :class:`Transcript`.

    Parameters
    ----------
    transcript:
        Source transcript.
    output_path:
        Destination ``.srt`` file path.

    Returns
    -------
    Path
        Path to the written SRT file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    srt_content = transcript.to_srt()
    output_path.write_text(srt_content, encoding="utf-8")
    logger.info("Generated SRT: %s (%d segments)", output_path, len(transcript.segments))
    return output_path


def generate_ass(
    transcript: Transcript,
    style_name: str,
    output_path: Path,
) -> Path:
    """
    Write an ASS subtitle file with the specified caption style.

    Supported *style_name* values:
    - ``"dynamic"`` — large, bold, yellow karaoke-style word highlighting
    - ``"clean"``   — white text, centered, standard
    - ``"minimal"`` — small text, translucent background at the bottom

    Parameters
    ----------
    transcript:
        Source transcript.
    style_name:
        Caption style preset name.
    output_path:
        Destination ``.ass`` file path.

    Returns
    -------
    Path
        Path to the written ASS file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ass_content = transcript.to_ass(style=style_name)
    output_path.write_text(ass_content, encoding="utf-8")
    logger.info("Generated ASS captions: %s (style=%s)", output_path, style_name)
    return output_path
