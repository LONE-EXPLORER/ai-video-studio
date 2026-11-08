"""
app/editing/sidechain_ducker.py — Dynamic sidechain compression ducking.

Uses FFmpeg's sidechaincompress filter to automatically duck background music
whenever dialogue/speech is detected, creating broadcast-quality audio mix.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)

FFMPEG = "/opt/homebrew/bin/ffmpeg"


@dataclass
class DuckingParams:
    """Parameters for sidechain compression ducking."""
    # Attack time: how fast the ducker clamps down (ms)
    attack_ms: float = 5.0
    # Release time: how fast music returns after speech ends (ms)
    release_ms: float = 250.0
    # Threshold: dialogue level above which ducker activates (dB, negative)
    threshold_db: float = -24.0
    # Ratio: compression ratio when ducking (e.g. 10:1)
    ratio: float = 8.0
    # Knee: smoothness of threshold transition (dB)
    knee_db: float = 2.5
    # Music volume before ducking (linear 0–1)
    music_pre_vol: float = 0.35
    # Minimum music volume during heavy speech (linear 0–1)
    music_duck_floor: float = 0.08


class SidechainDucker:
    """
    Applies sidechain-compressed music ducking to a video.

    Workflow:
    1. Extract dialogue audio as sidechain signal.
    2. Mix with music: music volume dynamically compressed by dialogue level.
    3. Output final audio-mixed video.

    Falls back to static volume mixing if sidechain fails.
    """

    def __init__(self, params: Optional[DuckingParams] = None) -> None:
        self._params = params or DuckingParams()

    def duck_and_mix(
        self,
        dialogue_video: Path,
        music_path: Path,
        output_path: Path,
        music_volume: float = 0.35,
    ) -> Path:
        """
        Apply sidechain ducking and produce final mixed video.

        Parameters
        ----------
        dialogue_video:
            Input video with dialogue audio track.
        music_path:
            Background music file (WAV/MP3/AAC).
        output_path:
            Destination for mixed output video.
        music_volume:
            Base music volume before ducking (0.0–1.0).

        Returns
        -------
        Path
            Output path (dialogue_video if mix failed).
        """
        p = self._params
        p.music_pre_vol = music_volume

        logger.info(
            "Applying sidechain ducking: attack=%.0fms release=%.0fms ratio=%.0f:1",
            p.attack_ms, p.release_ms, p.ratio,
        )

        # Build sidechaincompress filter
        # Input 0: dialogue video | Input 1: music
        # [0:a] = dialogue (sidechain signal) | [1:a] = music (compressed)
        threshold = 10 ** (p.threshold_db / 20.0)  # convert dB to linear
        
        filter_complex = (
            # Normalize dialogue for sidechain signal (not output)
            f"[0:a]aformat=sample_fmts=fltp,volume=1.5[sc];"
            # Prep music with pre-gain
            f"[1:a]aformat=sample_fmts=fltp,volume={p.music_pre_vol:.3f}[music_in];"
            # Apply sidechaincompress: music is compressed when dialogue is loud
            f"[music_in][sc]sidechaincompress="
            f"threshold={threshold:.4f}:"
            f"ratio={p.ratio:.1f}:"
            f"attack={p.attack_ms:.1f}:"
            f"release={p.release_ms:.1f}:"
            f"knee={p.knee_db:.1f}:"
            f"level_sc=1.0:"
            f"makeup=1.0[music_ducked];"
            # Mix dialogue + ducked music
            f"[0:a][music_ducked]amix=inputs=2:duration=first:normalize=0,"
            f"loudnorm=I=-16:TP=-1.5:LRA=11,"
            f"aformat=sample_fmts=s16,aresample=48000[audio_out]"
        )

        cmd = [
            FFMPEG, "-y",
            "-i", str(dialogue_video),
            "-stream_loop", "-1", "-i", str(music_path),
            "-filter_complex", filter_complex,
            "-map", "0:v",
            "-map", "[audio_out]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-ac", "2",
            "-shortest",
            str(output_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if result.returncode == 0:
                logger.info("Sidechain ducking complete → %s", output_path)
                return output_path
            else:
                logger.warning(
                    "Sidechain ducking failed (rc=%d); falling back to static mix.\nFFmpeg: %s",
                    result.returncode, result.stderr[-500:],
                )
                return self._static_fallback(dialogue_video, music_path, output_path, music_volume)
        except subprocess.TimeoutExpired:
            logger.error("Sidechain ducking timed out; falling back to static mix")
            return self._static_fallback(dialogue_video, music_path, output_path, music_volume)

    def _static_fallback(
        self,
        dialogue_video: Path,
        music_path: Path,
        output_path: Path,
        music_volume: float,
    ) -> Path:
        """Simple volume-based music duck without sidechain compression."""
        logger.info("Static music mix fallback at vol=%.2f", music_volume)
        cmd = [
            FFMPEG, "-y",
            "-i", str(dialogue_video),
            "-stream_loop", "-1", "-i", str(music_path),
            "-filter_complex",
            f"[1:a]volume={music_volume:.3f}[music];"
            f"[0:a][music]amix=inputs=2:duration=first:normalize=0,"
            f"aformat=sample_fmts=s16[out]",
            "-map", "0:v",
            "-map", "[out]",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-ar", "48000", "-ac", "2",
            "-shortest",
            str(output_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                return output_path
        except subprocess.TimeoutExpired:
            logger.error("Static fallback timed out")
        # Last resort: return original without music
        logger.error("Static fallback also failed; returning original video")
        import shutil
        shutil.copy2(dialogue_video, output_path)
        return output_path
