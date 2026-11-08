"""
scripts/generate_test_video.py — Synthetic test video generator using FFmpeg.

Generates a valid MP4 video with:
- Video test pattern with animated timecode
- Audio tone with simulated speech cadence
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from app.utils.ffmpeg_utils import find_ffmpeg


def generate_test_video(
    output_path: Path,
    duration: float = 30.0,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
) -> Path:
    """
    Generate a synthetic test video file using FFmpeg.

    Parameters
    ----------
    output_path:
        Destination MP4 path.
    duration:
        Video duration in seconds.
    width:
        Frame width.
    height:
        Frame height.
    fps:
        Frames per second.

    Returns
    -------
    Path
        Path to the generated video.
    """
    ffmpeg_bin = find_ffmpeg()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate synthetic video with color bars and timecode, plus audio tone
    # Audio uses a 440Hz sine wave pulsing to simulate speech activity
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f", "lavfi",
        "-i", f"testsrc=size={width}x{height}:rate={fps}",
        "-f", "lavfi",
        "-i", f"sine=frequency=440:beep_factor=4:sample_rate=16000",
        "-t", str(duration),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        str(output_path),
    ]

    print(f"Generating synthetic test video: {output_path} ({width}x{height}, {duration}s)...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg failed to generate test video: {res.stderr}")

    print(f"Successfully generated test video: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic test video")
    parser.add_argument("--output", type=str, default="test_sample.mp4", help="Output path")
    parser.add_argument("--duration", type=float, default=20.0, help="Duration in seconds")
    parser.add_argument("--vertical", action="store_true", help="Generate 9:16 vertical video (1080x1920)")
    args = parser.parse_args()

    w, h = (1080, 1920) if args.vertical else (1920, 1080)
    generate_test_video(Path(args.output), duration=args.duration, width=w, height=h)
