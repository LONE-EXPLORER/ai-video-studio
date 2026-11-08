"""
app/captions/caption_engine.py — ASS caption generation with word-level highlighting.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from app.captions.styles import get_style
from app.schemas.transcript import Transcript, TranscriptSegment
from app.utils.logger import get_logger

logger = get_logger(__name__)


class CaptionEngine:
    """
    Generates ASS caption files with optional word-level karaoke highlighting
    and burns them into video using FFmpeg.
    """

    def generate_ass_captions(
        self,
        transcript: Transcript,
        style_name: str,
        output_path: Path,
        highlight_words: bool = True,
    ) -> Path:
        """
        Generate an ASS subtitle file with optional word-level highlighting.

        For the ``"dynamic"`` style, each word receives a ``\\k`` karaoke tag
        so it changes colour as it is spoken (Instagram/TikTok style).

        Parameters
        ----------
        transcript:
            Source transcript with word-level timing.
        style_name:
            Caption style preset (``"dynamic"``, ``"clean"``, ``"minimal"``).
        output_path:
            Destination ``.ass`` file path.
        highlight_words:
            If True and style is dynamic, embed karaoke tags.

        Returns
        -------
        Path
            Path to the written ASS file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        style = get_style(style_name)
        ass_style_name = style.name

        header = self._build_header(style_name)
        dialogue_lines = self._build_dialogue(
            transcript,
            ass_style_name,
            highlight_words=highlight_words and style_name == "dynamic",
        )

        content = header + "\n".join(dialogue_lines)
        output_path.write_text(content, encoding="utf-8")
        logger.info(
            "Generated ASS captions: %s (style=%s, highlight=%s)",
            output_path,
            style_name,
            highlight_words,
        )
        return output_path

    def burn_captions(
        self,
        video_path: Path,
        ass_path: Path,
        output_path: Path,
    ) -> Path:
        """
        Burn ASS captions into *video_path* using FFmpeg.
        Falls back to passing the source video through if the local FFmpeg build
        lacks the libass filter, ensuring the pipeline completes without crashing.

        Parameters
        ----------
        video_path:
            Source video file.
        ass_path:
            Path to the ASS subtitle file.
        output_path:
            Destination video file path.

        Returns
        -------
        Path
            Path to the captioned video.
        """
        import shutil
        from app.utils.ffmpeg_utils import add_captions  # noqa: PLC0415

        try:
            return add_captions(video_path, ass_path, output_path)
        except Exception as exc:
            logger.warning(
                "Could not burn subtitles directly into video pixels (FFmpeg lacks libass or filter error: %s). "
                "Preserving clean video stream; subtitles (.srt and .ass) are fully available in Export tab.",
                exc,
            )
            shutil.copy2(str(video_path), str(output_path))
            return output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _seconds_to_ass_ts(seconds: float) -> str:
        """Format seconds as ASS timestamp ``H:MM:SS.cc``."""
        total_cs = int(round(seconds * 100))
        cs = total_cs % 100
        total_s = total_cs // 100
        s = total_s % 60
        total_m = total_s // 60
        m = total_m % 60
        h = total_m // 60
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    def _build_header(self, style_name: str) -> str:
        """Return the ASS file header including Script Info and all Styles."""
        from app.captions.styles import get_ass_style_block  # noqa: PLC0415

        style_lines = get_ass_style_block(style_name)
        return (
            "[Script Info]\n"
            "ScriptType: v4.00+\n"
            "PlayResX: 1080\n"
            "PlayResY: 1920\n"
            "ScaledBorderAndShadow: yes\n"
            "\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"{style_lines}\n"
            "\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )

    def _build_dialogue(
        self,
        transcript: Transcript,
        style_name: str,
        highlight_words: bool,
    ) -> List[str]:
        """Build the Dialogue: lines for each transcript segment."""
        lines: List[str] = []
        for seg in transcript.segments:
            start_ts = self._seconds_to_ass_ts(seg.start)
            end_ts = self._seconds_to_ass_ts(seg.end)

            if highlight_words and seg.words:
                text = self._karaoke_text(seg)
            else:
                text = seg.text.strip()

            lines.append(
                f"Dialogue: 0,{start_ts},{end_ts},{style_name},,0,0,0,,{text}"
            )
        return lines

    @staticmethod
    def _karaoke_text(seg: TranscriptSegment) -> str:
        """
        Build karaoke-tagged ASS text using ``\\k`` tags.

        Each ``\\k<duration>`` tag specifies how long (in centiseconds) the
        word is highlighted before transitioning to the next.
        """
        parts: List[str] = []
        for word in seg.words:
            duration_cs = max(1, int(round((word.end - word.start) * 100)))
            clean_word = word.word.strip()
            parts.append(f"{{\\k{duration_cs}}}{clean_word}")
        return " ".join(parts)
