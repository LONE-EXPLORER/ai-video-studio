"""
app/editing/normalizer.py — MediaNormalizer for pre-conforming heterogeneous source videos.

Standardizes video geometry, aspect ratio, frame rate, color space, and audio layout
to guarantee seamless concatenation and prevent non-monotonic DTS / timestamp corruption.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional

from app.utils import ffmpeg_utils
from app.utils.logger import get_logger

logger = get_logger(__name__)


class MediaNormalizer:
    """
    Standardizes heterogeneous video assets to a canonical production format:
    - Dimensions: 1080x1920 (9:16 vertical, SAR 1:1) or custom dimensions
    - Frame rate: 30 fps Constant Frame Rate (CFR)
    - Color Primaries & Matrix: BT.709, TV range
    - Audio: 48,000 Hz Stereo AAC at 192 kbps (with synthetic silence injection if audio is missing)
    - Hardware acceleration: Apple Silicon VideoToolbox with automatic libx264 fallback
    """

    def __init__(
        self,
        target_w: int = 1080,
        target_h: int = 1920,
        fps: int = 30,
        video_bitrate: str = "6000k",
        audio_bitrate: str = "192k",
    ) -> None:
        self.target_w = target_w
        self.target_h = target_h
        self.fps = fps
        self.video_bitrate = video_bitrate
        self.audio_bitrate = audio_bitrate

    def conform_clip(
        self,
        input_path: Path,
        output_path: Path,
        pad_mode: Literal["auto", "letterbox", "blur"] = "auto",
        target_w: Optional[int] = None,
        target_h: Optional[int] = None,
        fps: Optional[int] = None,
    ) -> Path:
        """
        Conform a single input video file to canonical standards.

        Parameters
        ----------
        input_path:
            Source video file path.
        output_path:
            Target conformed video file path.
        pad_mode:
            'letterbox': fit inside frame with black centered padding.
            'blur': split into blurred background and centered foreground (ideal for 16:9 -> 9:16).
            'auto': uses 'blur' if input is landscape and target is vertical, else 'letterbox'.
        target_w:
            Optional override for target width (defaults to self.target_w).
        target_h:
            Optional override for target height (defaults to self.target_h).
        fps:
            Optional override for framerate (defaults to self.fps).

        Returns
        -------
        Path
            Path to the conformed video.
        """
        input_p = Path(input_path).resolve()
        if not input_p.exists():
            raise FileNotFoundError(f"Source video file not found: {input_p}")

        output_p = Path(output_path).resolve()
        output_p.parent.mkdir(parents=True, exist_ok=True)

        w = target_w or self.target_w
        h = target_h or self.target_h
        rate = fps or self.fps

        probe = ffmpeg_utils.probe_video(input_p)
        vstream = next(
            (s for s in probe.get("streams", []) if s.get("codec_type") == "video"),
            {},
        )
        astream = next(
            (s for s in probe.get("streams", []) if s.get("codec_type") == "audio"),
            None,
        )

        in_w = int(vstream.get("width", w))
        in_h = int(vstream.get("height", h))
        has_audio = astream is not None

        fmt = probe.get("format", {})
        try:
            duration = float(fmt.get("duration", vstream.get("duration", 1.0)))
        except (ValueError, TypeError):
            duration = 1.0
        if duration <= 0:
            duration = 1.0

        # Resolve pad mode
        resolved_pad = pad_mode
        if resolved_pad == "auto":
            is_input_landscape = in_w > in_h
            is_target_vertical = w < h
            resolved_pad = "blur" if (is_input_landscape and is_target_vertical) else "letterbox"

        logger.info(
            "Conforming '%s' (%dx%d, audio=%s) -> '%s' (mode=%s, %dx%d @ %dfps)",
            input_p.name,
            in_w,
            in_h,
            has_audio,
            output_p.name,
            resolved_pad,
            w,
            h,
            rate,
        )

        ffmpeg_bin = ffmpeg_utils.find_ffmpeg()

        # Build video filtergraph
        if resolved_pad == "blur":
            vfilter = (
                f"[0:v]split=2[bg_in][fg_in];"
                f"[bg_in]scale={w}:{h}:force_original_aspect_ratio=increase,"
                f"crop={w}:{h},boxblur=25:5[bg];"
                f"[fg_in]scale={w}:{h}:force_original_aspect_ratio=decrease[fg];"
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,fps={rate}[v_norm]"
            )
        else:
            vfilter = (
                f"[0:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,"
                f"setsar=1,fps={rate}[v_norm]"
            )

        if has_audio:
            afilter = "[0:a]aresample=48000,aformat=channel_layouts=stereo[a_norm]"
            filter_complex = f"{vfilter};{afilter}"
        else:
            # Inject synthetic silence to guarantee an audio stream
            filter_complex = (
                f"{vfilter};"
                f"anullsrc=r=48000:cl=stereo,atrim=duration={duration:.3f}[a_norm]"
            )

        map_args = ["-map", "[v_norm]", "-map", "[a_norm]"]

        # Strategy 1: Attempt Apple Silicon VideoToolbox hardware acceleration
        vt_args = [
            ffmpeg_bin, "-y",
            "-i", str(input_p),
            "-filter_complex", filter_complex,
            *map_args,
            "-c:v", "h264_videotoolbox",
            "-b:v", self.video_bitrate,
            "-maxrate", "8000k",
            "-pix_fmt", "nv12",
            "-allow_sw", "1",
            "-color_range", "tv",
            "-colorspace", "bt709",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-c:a", "aac",
            "-b:a", self.audio_bitrate,
            "-ar", "48000",
            "-ac", "2",
            "-movflags", "+faststart",
            str(output_p),
        ]

        try:
            ffmpeg_utils._run(vt_args)
            logger.info("Conformed video successfully via VideoToolbox -> %s", output_p)
            return output_p
        except RuntimeError as exc:
            logger.warning("VideoToolbox normalizer failed (%s), falling back to libx264", exc)

        # Strategy 2: Fallback to software libx264
        sw_args = [
            ffmpeg_bin, "-y",
            "-i", str(input_p),
            "-filter_complex", filter_complex,
            *map_args,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-color_range", "tv",
            "-colorspace", "bt709",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-c:a", "aac",
            "-b:a", self.audio_bitrate,
            "-ar", "48000",
            "-ac", "2",
            "-movflags", "+faststart",
            str(output_p),
        ]
        ffmpeg_utils._run(sw_args)
        logger.info("Conformed video successfully via libx264 -> %s", output_p)
        return output_p

    def conform_sources(
        self,
        source_paths: List[Path],
        output_dir: Path,
        pad_mode: Literal["auto", "letterbox", "blur"] = "auto",
        target_w: Optional[int] = None,
        target_h: Optional[int] = None,
        fps: Optional[int] = None,
    ) -> Dict[str, Path]:
        """
        Conform a list of source videos and map them to indexed source IDs:
        {"source_0": Path(".../conformed_source_0.mp4"), "source_1": ...}
        """
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        source_map: Dict[str, Path] = {}

        for idx, src in enumerate(source_paths):
            src_id = f"source_{idx}"
            out_file = output_dir / f"conformed_{src_id}.mp4"
            self.conform_clip(
                Path(src),
                out_file,
                pad_mode=pad_mode,
                target_w=target_w,
                target_h=target_h,
                fps=fps,
            )
            source_map[src_id] = out_file

        return source_map
