"""
app/schemas/video_meta.py — VideoMetadata Pydantic model.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class VideoStream(BaseModel):
    """Parsed video stream information from ffprobe."""

    codec_name: str
    width: Optional[int] = None
    height: Optional[int] = None
    r_frame_rate: str
    duration: Optional[float] = None
    bit_rate: Optional[int] = None
    pix_fmt: Optional[str] = None


class AudioStream(BaseModel):
    """Parsed audio stream information from ffprobe."""

    codec_name: str
    sample_rate: int
    channels: int
    duration: Optional[float] = None
    bit_rate: Optional[int] = None


class VideoMetadata(BaseModel):
    """Complete video metadata derived from ffprobe output."""

    file_path: str
    file_size_mb: float
    duration_seconds: float
    video_stream: VideoStream
    audio_stream: Optional[AudioStream] = None
    format_name: str
    has_audio: bool
    frame_rate: float  # parsed from r_frame_rate fraction string

    # ------------------------------------------------------------------
    # Factory method
    # ------------------------------------------------------------------

    @classmethod
    def from_ffprobe(cls, probe_data: dict, file_path: str) -> "VideoMetadata":
        """
        Parse ffprobe JSON output into a :class:`VideoMetadata` instance.

        Parameters
        ----------
        probe_data:
            Parsed JSON dict returned by :func:`app.utils.ffmpeg_utils.probe_video`.
        file_path:
            Absolute path string of the source file.

        Returns
        -------
        VideoMetadata
        """
        from pathlib import Path as _Path

        streams = probe_data.get("streams", [])
        fmt = probe_data.get("format", {})

        # Locate video and audio streams
        video_stream_data: Optional[dict] = None
        audio_stream_data: Optional[dict] = None
        for s in streams:
            codec_type = s.get("codec_type", "")
            if codec_type == "video" and video_stream_data is None:
                video_stream_data = s
            elif codec_type == "audio" and audio_stream_data is None:
                audio_stream_data = s

        if video_stream_data is None:
            raise ValueError(f"No video stream found in {file_path}")

        # --- Parse frame rate ---
        r_frame_rate = video_stream_data.get("r_frame_rate", "25/1")
        try:
            num, den = r_frame_rate.split("/")
            frame_rate = float(num) / float(den) if float(den) != 0 else 0.0
        except (ValueError, ZeroDivisionError):
            frame_rate = 0.0

        # --- Duration ---
        duration = float(
            video_stream_data.get("duration")
            or fmt.get("duration")
            or 0.0
        )

        # --- File size ---
        raw_path = _Path(file_path)
        try:
            file_size_mb = raw_path.stat().st_size / (1024 * 1024)
        except FileNotFoundError:
            file_size_mb = float(fmt.get("size", 0)) / (1024 * 1024)

        # --- Video stream model ---
        video_stream = VideoStream(
            codec_name=video_stream_data.get("codec_name", "unknown"),
            width=video_stream_data.get("width"),
            height=video_stream_data.get("height"),
            r_frame_rate=r_frame_rate,
            duration=float(video_stream_data["duration"]) if "duration" in video_stream_data else None,
            bit_rate=int(video_stream_data["bit_rate"]) if "bit_rate" in video_stream_data else None,
            pix_fmt=video_stream_data.get("pix_fmt"),
        )

        # --- Audio stream model ---
        audio_stream: Optional[AudioStream] = None
        if audio_stream_data:
            audio_stream = AudioStream(
                codec_name=audio_stream_data.get("codec_name", "unknown"),
                sample_rate=int(audio_stream_data.get("sample_rate", 44100)),
                channels=int(audio_stream_data.get("channels", 2)),
                duration=float(audio_stream_data["duration"]) if "duration" in audio_stream_data else None,
                bit_rate=int(audio_stream_data["bit_rate"]) if "bit_rate" in audio_stream_data else None,
            )

        return cls(
            file_path=file_path,
            file_size_mb=round(file_size_mb, 2),
            duration_seconds=round(duration, 3),
            video_stream=video_stream,
            audio_stream=audio_stream,
            format_name=fmt.get("format_name", "unknown"),
            has_audio=audio_stream is not None,
            frame_rate=round(frame_rate, 4),
        )
