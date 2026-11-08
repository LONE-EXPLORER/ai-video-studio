"""
app/rendering/broll_compositor.py — Hardware-Accelerated B-Roll Overlay Compositor.

Applies Apple Silicon VideoToolbox hardware-accelerated FFmpeg compositing to overlay
B-roll cutaway video clips onto the base timeline. Enforces PTS alignment, input looping,
9:16 aspect ratio conformation, audio track 0 preservation, and automatic fallback to libx264.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

from app.schemas.editing_plan import BrollOverlay
from app.utils import ffmpeg_utils
from app.utils.logger import get_logger

logger = get_logger(__name__)


class BrollCompositor:
    """
    Composites secondary B-roll cutaway overlays onto the primary timeline.
    """

    def __init__(self, ffmpeg_bin: Optional[str] = None) -> None:
        self.ffmpeg_bin = ffmpeg_bin or ffmpeg_utils.find_ffmpeg()
        self._vt_available = self._check_videotoolbox()

    def _check_videotoolbox(self) -> bool:
        """Check if h264_videotoolbox is available in FFmpeg."""
        try:
            res = subprocess.run(
                [self.ffmpeg_bin, "-encoders"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return "h264_videotoolbox" in res.stdout
        except Exception:
            return False

    def composite_overlays(
        self,
        base_video_path: Path,
        overlays: List[BrollOverlay],
        output_path: Path,
        target_w: int = 1080,
        target_h: int = 1920,
        fps: int = 30,
        video_bitrate: str = "6000k",
    ) -> Path:
        """
        Overlay B-roll cutaways onto base video with PTS shifting and VideoToolbox encoding.

        Parameters
        ----------
        base_video_path:
            Path to the base speaker video (rough cut conformed to target resolution).
        overlays:
            List of BrollOverlay instances specifying timeline intervals and source media.
        output_path:
            Destination path for the composited video.
        target_w:
            Target width in pixels (default 1080).
        target_h:
            Target height in pixels (default 1920).
        fps:
            Target frame rate (default 30).
        video_bitrate:
            Target video encoding bitrate (default "6000k").

        Returns
        -------
        Path
            Path to output video (or base_video_path if no valid overlays to composite).
        """
        if not base_video_path.exists():
            raise FileNotFoundError(f"Base video path does not exist: {base_video_path}")

        # Probe base video duration and audio stream presence
        base_probe = ffmpeg_utils.probe_video(base_video_path)
        base_duration = float(base_probe.get("format", {}).get("duration", 0.0))
        if base_duration <= 0.0:
            for s in base_probe.get("streams", []):
                if s.get("codec_type") == "video" and "duration" in s:
                    base_duration = float(s["duration"])
                    break

        has_audio = any(s.get("codec_type") == "audio" for s in base_probe.get("streams", []))

        # Filter and sanitize valid overlays
        valid_overlays: List[BrollOverlay] = []
        for o in overlays:
            src = Path(o.source_path)
            if not src.exists() or src.stat().st_size == 0:
                logger.warning("B-roll source file '%s' not found or empty; skipping overlay %s", src, o.broll_id)
                continue

            # Drop overlays starting at or beyond video end
            if base_duration > 0 and o.timeline_start >= base_duration:
                logger.warning(
                    "B-roll %s starts at %.2fs beyond base duration %.2fs; skipping",
                    o.broll_id, o.timeline_start, base_duration,
                )
                continue

            # Clamp overlay end to base duration
            clamped_end = min(o.timeline_end, base_duration) if base_duration > 0 else o.timeline_end
            if clamped_end <= o.timeline_start:
                continue

            # Copy with clamped end
            valid_overlays.append(BrollOverlay(
                broll_id=o.broll_id,
                timeline_start=o.timeline_start,
                timeline_end=round(clamped_end, 3),
                source_path=str(src.resolve()),
                overlay_type=o.overlay_type,
            ))

        if not valid_overlays:
            logger.info("No valid B-roll overlays to composite. Returning base video.")
            if output_path != base_video_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(base_video_path, output_path)
                return output_path
            return base_video_path

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._execute_ffmpeg_composite(
                base_video_path=base_video_path,
                overlays=valid_overlays,
                output_path=output_path,
                target_w=target_w,
                target_h=target_h,
                fps=fps,
                video_bitrate=video_bitrate,
                has_audio=has_audio,
                use_videotoolbox=self._vt_available,
            )
        except Exception as exc:
            if self._vt_available:
                logger.warning("VideoToolbox compositing failed (%s); falling back to libx264", exc)
                self._execute_ffmpeg_composite(
                    base_video_path=base_video_path,
                    overlays=valid_overlays,
                    output_path=output_path,
                    target_w=target_w,
                    target_h=target_h,
                    fps=fps,
                    video_bitrate=video_bitrate,
                    has_audio=has_audio,
                    use_videotoolbox=False,
                )
            else:
                raise

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(f"Compositing failed: output file '{output_path}' was not generated.")

        logger.info(
            "Successfully composited %d B-roll cutaway(s) -> %s",
            len(valid_overlays),
            output_path.name,
        )
        return output_path

    def _execute_ffmpeg_composite(
        self,
        base_video_path: Path,
        overlays: List[BrollOverlay],
        output_path: Path,
        target_w: int,
        target_h: int,
        fps: int,
        video_bitrate: str,
        has_audio: bool,
        use_videotoolbox: bool,
    ) -> None:
        """Construct and execute FFmpeg filtergraph for chained B-roll overlays."""
        cmd: List[str] = [self.ffmpeg_bin, "-y", "-i", str(base_video_path)]

        # Add looped B-roll inputs
        for o in overlays:
            cmd.extend(["-stream_loop", "-1", "-i", o.source_path])

        filter_chains: List[str] = []
        prev_v = "0:v"

        for idx, o in enumerate(overlays):
            input_idx = idx + 1
            b_label = f"b{idx}"
            next_v = f"v{idx}" if idx < len(overlays) - 1 else "vout"
            start = o.timeline_start
            end = o.timeline_end

            # Conformation + critical PTS shift formula
            if o.overlay_type == "pip":
                # Centered picture-in-picture in upper frame
                filter_chains.append(
                    f"[{input_idx}:v]scale=864:486:force_original_aspect_ratio=decrease,"
                    f"setsar=1,fps={fps},setpts=PTS-STARTPTS+({start:.4f}/TB)[{b_label}]"
                )
                overlay_pos = "(W-w)/2:(H-h)*0.35"
            else:
                # Default fullscreen cutaway conforming to 9:16 target_w:target_h
                filter_chains.append(
                    f"[{input_idx}:v]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
                    f"crop={target_w}:{target_h},setsar=1,fps={fps},"
                    f"setpts=PTS-STARTPTS+({start:.4f}/TB)[{b_label}]"
                )
                overlay_pos = "0:0"

            filter_chains.append(
                f"[{prev_v}][{b_label}]overlay={overlay_pos}:"
                f"enable='between(t,{start:.4f},{end:.4f})':eof_action=pass[{next_v}]"
            )
            prev_v = next_v

        filter_complex = ";".join(filter_chains)

        encoder_args = (
            [
                "-c:v", "h264_videotoolbox",
                "-b:v", video_bitrate,
                "-maxrate", "8000k",
                "-pix_fmt", "nv12",
                "-color_range", "tv",
                "-colorspace", "bt709",
                "-color_primaries", "bt709",
                "-color_trc", "bt709",
            ]
            if use_videotoolbox
            else [
                "-c:v", "libx264",
                "-crf", "20",
                "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-color_range", "tv",
                "-colorspace", "bt709",
            ]
        )

        audio_args: List[str] = []
        if has_audio:
            # Preserve base video dialogue audio track 0 completely intact
            audio_args = ["-map", "0:a", "-c:a", "copy"]
        else:
            audio_args = ["-an"]

        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", f"[{prev_v}]",
            *audio_args,
            *encoder_args,
            "-movflags", "+faststart",
            str(output_path),
        ])

        try:
            ffmpeg_utils._run(cmd)
        except RuntimeError as exc:
            # If audio copy failed due to container remux issues, retry with AAC re-encoding
            if has_audio and "copy" in audio_args:
                logger.warning("Audio stream copy failed during compositing; retrying with AAC encoder: %s", exc)
                cmd_aac = list(cmd)
                idx_ca = cmd_aac.index("-c:a")
                cmd_aac[idx_ca + 1] = "aac"
                cmd_aac.insert(idx_ca + 2, "-b:a")
                cmd_aac.insert(idx_ca + 3, "192k")
                ffmpeg_utils._run(cmd_aac)
            else:
                raise
