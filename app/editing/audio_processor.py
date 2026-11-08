"""
app/editing/audio_processor.py — Audio normalisation and mixing utilities.
"""

from __future__ import annotations

from pathlib import Path

from app.utils.ffmpeg_utils import find_ffmpeg, _run
from app.utils.logger import get_logger

logger = get_logger(__name__)


def normalize_audio_loudnorm(
    input_path: Path,
    output_path: Path,
    target_lufs: float = -16.0,
) -> Path:
    """
    Normalise audio to a target integrated loudness using FFmpeg's loudnorm filter.

    Uses a two-pass approach: first pass analyses the file, second pass applies
    correction. For simplicity, the linear normalisation is applied in one pass.

    Parameters
    ----------
    input_path:
        Source video or audio file.
    output_path:
        Destination file path.
    target_lufs:
        Target integrated loudness in LUFS (default -16 for social media).

    Returns
    -------
    Path
        Path to the normalised output file.
    """
    ffmpeg = find_ffmpeg()
    args = [
        ffmpeg, "-y",
        "-i", str(input_path),
        "-af", f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:print_format=summary",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-ac", "2",
        str(output_path),
    ]
    _run(args)
    logger.info("Normalised audio (target=%.1f LUFS) → %s", target_lufs, output_path)
    return output_path


def duck_music(
    dialogue_path: Path,
    music_path: Path,
    output_path: Path,
    music_vol: float = 0.15,
) -> Path:
    """
    Mix dialogue audio with background music at a reduced volume using ``amix``.

    The dialogue track plays at full volume; the music track is attenuated by
    *music_vol* (0.0–1.0).

    Parameters
    ----------
    dialogue_path:
        Source video/audio file containing the dialogue track.
    music_path:
        Background music audio file.
    output_path:
        Destination file path.
    music_vol:
        Volume multiplier for the music track (default 0.15 = 15 %).

    Returns
    -------
    Path
        Path to the mixed output file.
    """
    ffmpeg = find_ffmpeg()

    # Build filter: attenuate music before mixing
    # Input 0 = dialogue, Input 1 = music
    af = (
        f"[1:a]volume={music_vol}[music];"
        f"[0:a][music]amix=inputs=2:duration=first:dropout_transition=3[aout]"
    )

    args = [
        ffmpeg, "-y",
        "-i", str(dialogue_path),
        "-stream_loop", "-1",  # loop music if shorter than dialogue
        "-i", str(music_path),
        "-filter_complex", af,
        "-map", "0:v?",         # keep video if present
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-ac", "2",
        "-shortest",
        str(output_path),
    ]
    _run(args)
    logger.info("Mixed music (vol=%.2f) → %s", music_vol, output_path)
    return output_path
