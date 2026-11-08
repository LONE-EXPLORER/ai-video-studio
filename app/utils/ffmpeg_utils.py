"""
app/utils/ffmpeg_utils.py — Safe FFmpeg/FFprobe subprocess wrappers.

All subprocess calls use argument arrays (never shell=True).
FFmpeg errors surface the actual stderr in the exception message.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------

_EXTRA_PATHS = [
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/usr/bin",
]


def _find_binary(name: str) -> str:
    """
    Locate a binary by checking PATH first, then common macOS install paths.

    Parameters
    ----------
    name:
        Binary name, e.g. ``"ffmpeg"``.

    Returns
    -------
    str
        Absolute path to the binary.

    Raises
    ------
    RuntimeError
        If the binary cannot be found.
    """
    found = shutil.which(name)
    if found:
        return found

    for base in _EXTRA_PATHS:
        candidate = Path(base) / name
        if candidate.is_file() and shutil.which(str(candidate)):
            return str(candidate)
        # Even if shutil.which fails, check if file is executable
        if candidate.is_file():
            import os
            if os.access(candidate, os.X_OK):
                return str(candidate)

    raise RuntimeError(
        f"'{name}' not found. Install it via Homebrew: brew install ffmpeg"
    )


def find_ffmpeg() -> str:
    """Return the absolute path to the ``ffmpeg`` binary."""
    return _find_binary("ffmpeg")


def find_ffprobe() -> str:
    """Return the absolute path to the ``ffprobe`` binary."""
    return _find_binary("ffprobe")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run(args: List[str], timeout: int = 600) -> subprocess.CompletedProcess:
    """
    Execute a subprocess, logging the command at DEBUG level and raising
    RuntimeError (with stderr) on non-zero exit.
    """
    logger.debug("Running command: %s", " ".join(args))
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}):\n"
            f"CMD: {' '.join(args)}\n"
            f"STDERR: {result.stderr}"
        )
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def probe_video(path: Path) -> dict:
    """
    Run ``ffprobe`` on *path* and return the parsed JSON output.

    Parameters
    ----------
    path:
        Path to the video file.

    Returns
    -------
    dict
        Parsed ffprobe JSON containing ``streams`` and ``format`` keys.
    """
    ffprobe = find_ffprobe()
    args = [
        ffprobe,
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    result = _run(args)
    return json.loads(result.stdout)


def extract_audio(
    input_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
) -> Path:
    """
    Extract audio from *input_path* as a 16 kHz mono WAV file.

    Parameters
    ----------
    input_path:
        Source video file.
    output_path:
        Destination WAV path.
    sample_rate:
        Audio sample rate in Hz (default 16 000 for Whisper).

    Returns
    -------
    Path
        Path to the extracted WAV file.
    """
    ffmpeg = find_ffmpeg()
    args = [
        ffmpeg, "-y",
        "-i", str(input_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        str(output_path),
    ]
    _run(args)
    logger.info("Extracted audio to %s", output_path)
    return output_path


def extract_frames(
    input_path: Path,
    output_dir: Path,
    fps: float = 1.0,
) -> List[Path]:
    """
    Extract frames from *input_path* at *fps* frames per second.

    Parameters
    ----------
    input_path:
        Source video file.
    output_dir:
        Directory in which JPEG frames will be saved.
    fps:
        Frame extraction rate (default 1 fps).

    Returns
    -------
    list[Path]
        Sorted list of extracted JPEG paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()
    pattern = str(output_dir / "frame_%06d.jpg")
    args = [
        ffmpeg, "-y",
        "-i", str(input_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        pattern,
    ]
    _run(args)
    frames = sorted(output_dir.glob("frame_*.jpg"))
    logger.info("Extracted %d frames to %s", len(frames), output_dir)
    return frames


def cut_segment(
    input_path: Path,
    start: float,
    end: float,
    output_path: Path,
    use_videotoolbox: bool = True,
    micro_fade: bool = True,
    fade_duration: float = 0.01,
) -> Path:
    """
    Cut a segment from *start* to *end* seconds with exact frame accuracy,
    standardized 48kHz stereo AAC audio, and optional micro audio fades (10ms)
    to eliminate zero-crossing boundary clicks.

    Parameters
    ----------
    input_path:
        Source video file.
    start:
        Start time in seconds.
    end:
        End time in seconds.
    output_path:
        Destination file path.
    use_videotoolbox:
        Attempt hardware-accelerated encoding first.
    micro_fade:
        If True (default), applies micro audio fade-in and fade-out to prevent pops.
    fade_duration:
        Fade duration in seconds (default 0.01s / 10ms), clamped to clip_duration / 2.

    Returns
    -------
    Path
        Path to the cut segment.
    """
    ffmpeg = find_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check for audio stream presence to safely apply afade
    af_args: List[str] = []
    if micro_fade and end > start:
        try:
            probe = probe_video(input_path)
            has_audio = any(s.get("codec_type") == "audio" for s in probe.get("streams", []))
        except Exception:
            has_audio = True

        if has_audio:
            clip_dur = end - start
            fade_d = min(fade_duration, clip_dur / 2.0)
            if fade_d >= 0.0005:
                fade_out_st = max(0.0, clip_dur - fade_d)
                af_args = [
                    "-af",
                    f"afade=t=in:st=0:d={fade_d:.4f},afade=t=out:st={fade_out_st:.4f}:d={fade_d:.4f}",
                ]

    # Attempt VideoToolbox for fast hardware encoding on Apple Silicon
    if use_videotoolbox:
        try:
            args = [
                ffmpeg, "-y",
                "-ss", str(start),
                "-to", str(end),
                "-i", str(input_path),
                *af_args,
                "-c:v", "h264_videotoolbox",
                "-b:v", "5000k",
                "-c:a", "aac",
                "-ar", "48000",
                "-ac", "2",
                "-b:a", "192k",
                "-avoid_negative_ts", "make_zero",
                "-reset_timestamps", "1",
                str(output_path),
            ]
            _run(args)
            logger.debug("Cut segment (videotoolbox) [%.2f–%.2f] → %s", start, end, output_path)
            return output_path
        except RuntimeError:
            logger.debug("VideoToolbox cut failed or unavailable; falling back to libx264")

    args = [
        ffmpeg, "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", str(input_path),
        *af_args,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-c:a", "aac",
        "-ar", "48000",
        "-ac", "2",
        "-b:a", "192k",
        "-avoid_negative_ts", "make_zero",
        "-reset_timestamps", "1",
        str(output_path),
    ]
    _run(args)
    logger.debug("Cut segment (libx264) [%.2f–%.2f] → %s", start, end, output_path)
    return output_path



def concat_clips(
    clip_paths: List[Path],
    output_path: Path,
    fast_copy: bool = False,
) -> Path:
    """
    Concatenate clips reliably without non-monotonic DTS errors or container corruption.

    Uses hardware-accelerated filter_complex concat for clean A/V synchronization,
    falling back to demuxer with audio/video timestamp sanitization.

    Parameters
    ----------
    clip_paths:
        Ordered list of video clip paths to concatenate.
    output_path:
        Destination file path.
    fast_copy:
        If True, attempts stream copy first, falling back to re-encoding if failed.

    Returns
    -------
    Path
        Path to the concatenated video.
    """
    if not clip_paths:
        raise ValueError("clip_paths must not be empty")

    if len(clip_paths) == 1:
        shutil.copy2(clip_paths[0], output_path)
        logger.info("Single clip copied → %s", output_path)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()

    # Optional Fast Path: Stream Copy
    if fast_copy:
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, prefix="concat_"
            ) as f:
                for clip in clip_paths:
                    escaped = str(clip.resolve()).replace("'", "'\\''")
                    f.write(f"file '{escaped}'\n")
                concat_file = Path(f.name)
            try:
                args = [
                    ffmpeg, "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", str(concat_file),
                    "-c", "copy",
                    "-fflags", "+genpts",
                    "-avoid_negative_ts", "make_zero",
                    str(output_path),
                ]
                _run(args)
                logger.info("Concatenated %d clips via stream copy → %s", len(clip_paths), output_path)
                return output_path
            finally:
                concat_file.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Stream copy concat failed (%s); falling back to robust concat", exc)

    # Check if first clip has an audio stream
    has_audio = True
    try:
        probe = probe_video(clip_paths[0])
        has_audio = any(s.get("codec_type") == "audio" for s in probe.get("streams", []))
    except Exception:
        has_audio = True

    # Strategy A: Filter-Complex Concat (up to 32 clips)
    if len(clip_paths) <= 32:
        try:
            inputs = []
            for p in clip_paths:
                inputs.extend(["-i", str(p.resolve())])

            n = len(clip_paths)
            if has_audio:
                filter_str = "".join(f"[{i}:v][{i}:a]" for i in range(n)) + f"concat=n={n}:v=1:a=1[outv][outa]"
                map_args = ["-map", "[outv]", "-map", "[outa]"]
                audio_args = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
            else:
                filter_str = "".join(f"[{i}:v]" for i in range(n)) + f"concat=n={n}:v=1:a=0[outv]"
                map_args = ["-map", "[outv]"]
                audio_args = []

            # Try VideoToolbox first
            try:
                args = [
                    ffmpeg, "-y",
                    *inputs,
                    "-filter_complex", filter_str,
                    *map_args,
                    "-c:v", "h264_videotoolbox",
                    "-b:v", "5000k",
                    "-allow_sw", "1",
                    "-pix_fmt", "nv12",
                    *audio_args,
                    str(output_path),
                ]
                _run(args)
                logger.info("Concatenated %d clips via filter_complex (VideoToolbox) → %s", n, output_path)
                return output_path
            except RuntimeError:
                logger.debug("VideoToolbox concat failed; falling back to libx264")

            args = [
                ffmpeg, "-y",
                *inputs,
                "-filter_complex", filter_str,
                *map_args,
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "20",
                "-pix_fmt", "yuv420p",
                *audio_args,
                str(output_path),
            ]
            _run(args)
            logger.info("Concatenated %d clips via filter_complex (libx264) → %s", n, output_path)
            return output_path
        except Exception as exc:
            logger.warning("filter_complex concat failed (%s); falling back to sanitized demuxer", exc)

    # Strategy B: Demuxer Concat with PTS & Resampling Sanitization
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, prefix="concat_") as f:
        for clip in clip_paths:
            escaped = str(clip.resolve()).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
        concat_file = Path(f.name)

    try:
        v_filter_args = ["-vf", "setpts=N/FRAME_RATE/TB"]
        a_filter_args = ["-af", "aresample=async=1000:min_hard_comp=0.100000:first_pts=0,asetpts=N/SR/TB"] if has_audio else []
        a_codec_args = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"] if has_audio else []

        # VideoToolbox attempt
        try:
            args = [
                ffmpeg, "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                *v_filter_args,
                *a_filter_args,
                "-c:v", "h264_videotoolbox",
                "-b:v", "5000k",
                "-allow_sw", "1",
                "-pix_fmt", "nv12",
                *a_codec_args,
                str(output_path),
            ]
            _run(args)
            logger.info("Concatenated %d clips via sanitized demuxer (VideoToolbox) → %s", len(clip_paths), output_path)
            return output_path
        except RuntimeError:
            pass

        args = [
            ffmpeg, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            *v_filter_args,
            *a_filter_args,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            *a_codec_args,
            str(output_path),
        ]
        _run(args)
        logger.info("Concatenated %d clips via sanitized demuxer (libx264) → %s", len(clip_paths), output_path)
        return output_path
    finally:
        concat_file.unlink(missing_ok=True)


def apply_zoom_effect(
    input_path: Path,
    output_path: Path,
    zoom_factor: float = 1.05,
    duration: Optional[float] = None,
    discrete: bool = True,
    use_videotoolbox: bool = True,
) -> Path:
    """
    Apply a punch-in zoom effect to a video clip.

    By default, applies a discrete jump punch-in (instantaneous static magnification
    centered at eye level, upscaled with high-quality Lanczos interpolation) using
    hardware-accelerated VideoToolbox encoding.

    Parameters
    ----------
    input_path:
        Source video file.
    output_path:
        Destination file path.
    zoom_factor:
        Zoom multiplier (e.g. 1.15 = 15% magnification).
    duration:
        Clip duration in seconds. If None, probed from input.
    discrete:
        If True (default), applies a discrete jump cut (static 1.15x crop + upscale).
        If False, applies continuous linear slow zoom via zoompan.
    use_videotoolbox:
        Attempt hardware-accelerated encoding first on macOS.

    Returns
    -------
    Path
        Path to the zoomed video.
    """
    if zoom_factor <= 1.001:
        shutil.copy2(input_path, output_path)
        return output_path

    ffmpeg = find_ffmpeg()
    probe = probe_video(input_path)
    if duration is None:
        fmt = probe.get("format", {})
        duration = float(fmt.get("duration", 10.0))

    vstream = next(
        (s for s in probe.get("streams", []) if s.get("codec_type") == "video"),
        {},
    )
    width = int(vstream.get("width", 1080))
    height = int(vstream.get("height", 1920))
    fps = 30

    if discrete:
        # High-quality discrete jump punch-in: even crop + Lanczos upscale
        vf = (
            f"crop=w='trunc(in_w/{zoom_factor:.4f}/2)*2':"
            f"h='trunc(in_h/{zoom_factor:.4f}/2)*2':"
            f"x='(in_w-out_w)/2':y='(in_h-out_h)/2',"
            f"scale={width}:{height}:flags=lanczos,setsar=1"
        )
    else:
        # Continuous slow zoom fallback
        total_frames = max(int(duration * fps), 1)
        z_step = (zoom_factor - 1.0) / total_frames
        z_expr = f"min(1.0+on*{z_step:.6f},{zoom_factor})"
        vf = (
            f"zoompan=z='{z_expr}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={width}x{height}:fps={fps}"
        )

    # Attempt VideoToolbox first
    if use_videotoolbox:
        try:
            args = [
                ffmpeg, "-y",
                "-i", str(input_path),
                "-vf", vf,
                "-c:v", "h264_videotoolbox",
                "-b:v", "5000k",
                "-allow_sw", "1",
                "-pix_fmt", "nv12",
                "-c:a", "copy",
                str(output_path),
            ]
            _run(args)
            logger.debug("Applied discrete punch-in (VideoToolbox, factor=%.2f) → %s", zoom_factor, output_path)
            return output_path
        except RuntimeError:
            logger.debug("VideoToolbox punch-in failed; falling back to libx264")

    # Software fallback
    args = [
        ffmpeg, "-y",
        "-i", str(input_path),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(output_path),
    ]
    _run(args)
    logger.debug("Applied discrete punch-in (libx264, factor=%.2f) → %s", zoom_factor, output_path)
    return output_path



def add_captions(
    input_path: Path,
    ass_path: Path,
    output_path: Path,
) -> Path:
    """
    Burn ASS subtitles into the video.

    Parameters
    ----------
    input_path:
        Source video file.
    ass_path:
        Path to the ASS subtitle file.
    output_path:
        Destination file path.

    Returns
    -------
    Path
        Path to the captioned video.
    """
    ffmpeg = find_ffmpeg()
    # Escape colons and backslashes in the ASS path for the filtergraph
    safe_ass = str(ass_path).replace("\\", "/").replace(":", "\\:")
    args = [
        ffmpeg, "-y",
        "-i", str(input_path),
        "-vf", f"ass={safe_ass}",
        "-c:v", "libx264",
        "-c:a", "copy",
        str(output_path),
    ]
    _run(args)
    logger.info("Burned captions → %s", output_path)
    return output_path


def normalize_audio(input_path: Path, output_path: Path) -> Path:
    """
    Apply the loudnorm filter to normalise audio levels.

    Parameters
    ----------
    input_path:
        Source video/audio file.
    output_path:
        Destination file path.

    Returns
    -------
    Path
        Path to the normalised file.
    """
    ffmpeg = find_ffmpeg()
    args = [
        ffmpeg, "-y",
        "-i", str(input_path),
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-ac", "2",
        str(output_path),
    ]
    _run(args)
    logger.debug("Normalised audio → %s", output_path)
    return output_path


def apply_audio_fade(
    input_path: Path,
    output_path: Path,
    fade_in: float = 0.3,
    fade_out: float = 0.5,
) -> Path:
    """
    Apply audio fade-in and fade-out to a video file.

    Parameters
    ----------
    input_path:
        Source video file.
    output_path:
        Destination file path.
    fade_in:
        Fade-in duration in seconds.
    fade_out:
        Fade-out duration in seconds.

    Returns
    -------
    Path
        Path to the processed file.
    """
    ffmpeg = find_ffmpeg()

    # Determine file duration to calculate fade-out start time
    probe = probe_video(input_path)
    duration = float(probe.get("format", {}).get("duration", 0.0))
    fade_out_start = max(0.0, duration - fade_out)

    af = f"afade=t=in:st=0:d={fade_in},afade=t=out:st={fade_out_start:.3f}:d={fade_out}"
    args = [
        ffmpeg, "-y",
        "-i", str(input_path),
        "-af", af,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-ac", "2",
        str(output_path),
    ]
    _run(args)
    logger.debug("Applied audio fades (in=%.1fs, out=%.1fs) → %s", fade_in, fade_out, output_path)
    return output_path


def apply_micro_audio_fade(
    input_path: Path,
    output_path: Path,
    fade_duration: float = 0.01,
) -> Path:
    """
    Apply a 10ms (or custom) micro audio fade-in and fade-out to prevent
    zero-crossing audio clicks and pops at clip boundaries.
    """
    return apply_audio_fade(
        input_path=input_path,
        output_path=output_path,
        fade_in=fade_duration,
        fade_out=fade_duration,
    )


def render_final(
    input_path: Path,
    output_path: Path,
    width: int,
    height: int,
    crf: int = 23,
) -> Path:
    """
    Render final H.264/AAC MP4 with VideoToolbox acceleration hint on Apple Silicon.

    Falls back to libx264 software encoding if VideoToolbox is unavailable.

    Parameters
    ----------
    input_path:
        Edited/processed video file.
    output_path:
        Destination MP4 path.
    width:
        Target width in pixels.
    height:
        Target height in pixels.
    crf:
        Constant Rate Factor for libx264 (lower = higher quality).

    Returns
    -------
    Path
        Path to the rendered file.
    """
    ffmpeg = find_ffmpeg()

    # Pad/scale to target resolution maintaining aspect ratio with black bars
    vf = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"

    # Try VideoToolbox first (Apple Silicon / macOS hardware encoder)
    try:
        args = [
            ffmpeg, "-y",
            "-i", str(input_path),
            "-vf", vf,
            "-c:v", "h264_videotoolbox",
            "-b:v", "5000k",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-ac", "2",
            "-movflags", "+faststart",
            str(output_path),
        ]
        _run(args)
        logger.info("Rendered with VideoToolbox → %s", output_path)
        return output_path
    except RuntimeError:
        logger.warning("VideoToolbox not available, falling back to libx264")

    args = [
        ffmpeg, "-y",
        "-i", str(input_path),
        "-vf", vf,
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "medium",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-ac", "2",
        "-movflags", "+faststart",
        str(output_path),
    ]
    _run(args)

    logger.info("Rendered with libx264 → %s", output_path)
    return output_path


def generate_preview(
    input_path: Path,
    output_path: Path,
    max_width: int = 854,
) -> Path:
    """
    Generate a lower-bitrate preview video (web streaming friendly).

    Parameters
    ----------
    input_path:
        Source video file.
    output_path:
        Destination preview MP4 path.
    max_width:
        Maximum width in pixels (height scales proportionally).

    Returns
    -------
    Path
        Path to the preview file.
    """
    ffmpeg = find_ffmpeg()
    vf = f"scale={max_width}:-2"
    args = [
        ffmpeg, "-y",
        "-i", str(input_path),
        "-vf", vf,
        "-c:v", "libx264",
        "-crf", "28",
        "-preset", "fast",
        "-b:v", "1500k",
        "-maxrate", "1500k",
        "-bufsize", "3000k",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(output_path),
    ]
    _run(args)
    logger.info("Generated preview → %s", output_path)
    return output_path
