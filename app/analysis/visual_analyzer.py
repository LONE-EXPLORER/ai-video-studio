"""
app/analysis/visual_analyzer.py — Frame extraction and scene detection using FFmpeg.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import List

from app.utils.ffmpeg_utils import find_ffmpeg, find_ffprobe, _run
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VisualAnalyzer:
    """
    Analyses video content using FFmpeg-based filters.

    Provides:
    - Scene change detection
    - Single frame thumbnail extraction
    """

    def detect_scene_changes(
        self,
        video_path: Path,
        threshold: float = 0.4,
    ) -> List[float]:
        """
        Detect scene changes in *video_path* using FFmpeg's ``select`` filter.

        Parameters
        ----------
        video_path:
            Source video file.
        threshold:
            Scene change score threshold (0–1). Higher values detect only large cuts.

        Returns
        -------
        list[float]
            Sorted list of timestamps (in seconds) where scene changes occur.
        """
        ffmpeg = find_ffmpeg()
        args = [
            ffmpeg,
            "-i", str(video_path),
            "-vf", f"select=gt(scene\\,{threshold}),showinfo",
            "-vsync", "vfr",
            "-f", "null",
            "-",
        ]

        logger.debug("Detecting scene changes: threshold=%.2f", threshold)
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=600,
        )
        # showinfo writes to stderr; non-zero exit is expected (null muxer)
        stderr = result.stderr

        timestamps: List[float] = []
        # Parse lines like: [Parsed_showinfo_1 @ 0x...] n:  5 pts:  5000 pts_time:5.000 ...
        for match in re.finditer(r"pts_time:([\d.]+)", stderr):
            ts = float(match.group(1))
            timestamps.append(ts)

        timestamps.sort()
        logger.info("Detected %d scene changes in %s", len(timestamps), video_path.name)
        return timestamps

    def get_video_thumbnail(
        self,
        video_path: Path,
        timestamp: float,
        output_path: Path,
    ) -> Path:
        """
        Extract a single JPEG frame from *video_path* at *timestamp* seconds.

        Parameters
        ----------
        video_path:
            Source video file.
        timestamp:
            Time offset in seconds at which to capture the frame.
        output_path:
            Destination JPEG file path.

        Returns
        -------
        Path
            Path to the saved JPEG image.
        """
        ffmpeg = find_ffmpeg()
        args = [
            ffmpeg, "-y",
            "-ss", str(timestamp),
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",
            str(output_path),
        ]
        _run(args)
        logger.debug("Extracted thumbnail at %.2fs → %s", timestamp, output_path)
        return output_path
