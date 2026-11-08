"""
app/schemas/transcript.py — TranscriptWord, TranscriptSegment, and Transcript models.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class TranscriptWord(BaseModel):
    """A single word in the transcript with timing information."""

    word: str
    start: float  # seconds
    end: float    # seconds
    probability: Optional[float] = None


class TranscriptSegment(BaseModel):
    """A contiguous segment of spoken text, optionally with word-level timing."""

    id: int
    start: float
    end: float
    text: str
    words: List[TranscriptWord] = []


class Transcript(BaseModel):
    """Full transcript produced by the speech recognition engine."""

    language: str
    duration: float
    segments: List[TranscriptSegment]

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def full_text(self) -> str:
        """Return all segment text joined by spaces."""
        return " ".join(s.text.strip() for s in self.segments)

    def words_in_range(self, start: float, end: float) -> List[TranscriptWord]:
        """
        Return all words whose start time falls within [start, end].

        Parameters
        ----------
        start:
            Range start in seconds (inclusive).
        end:
            Range end in seconds (inclusive).

        Returns
        -------
        list[TranscriptWord]
        """
        result: List[TranscriptWord] = []
        for seg in self.segments:
            for word in seg.words:
                if start <= word.start <= end:
                    result.append(word)
        return result

    # ------------------------------------------------------------------
    # SRT / ASS export
    # ------------------------------------------------------------------

    @staticmethod
    def _seconds_to_srt_ts(seconds: float) -> str:
        """Convert seconds to SRT timestamp format ``HH:MM:SS,mmm``."""
        total_ms = int(round(seconds * 1000))
        ms = total_ms % 1000
        total_s = total_ms // 1000
        s = total_s % 60
        total_m = total_s // 60
        m = total_m % 60
        h = total_m // 60
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    @staticmethod
    def _seconds_to_ass_ts(seconds: float) -> str:
        """Convert seconds to ASS timestamp format ``H:MM:SS.cc``."""
        total_cs = int(round(seconds * 100))
        cs = total_cs % 100
        total_s = total_cs // 100
        s = total_s % 60
        total_m = total_s // 60
        m = total_m % 60
        h = total_m // 60
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

    def to_srt(self) -> str:
        """
        Generate an SRT subtitle string from all segments.

        Returns
        -------
        str
            Full SRT content.
        """
        lines: List[str] = []
        for i, seg in enumerate(self.segments, start=1):
            start_ts = self._seconds_to_srt_ts(seg.start)
            end_ts = self._seconds_to_srt_ts(seg.end)
            lines.append(str(i))
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(seg.text.strip())
            lines.append("")
        return "\n".join(lines)

    def to_ass(self, style: str = "default") -> str:
        """
        Generate an ASS subtitle string with optional word-level karaoke tags.

        Parameters
        ----------
        style:
            One of ``"default"``, ``"dynamic"``, ``"clean"``, ``"minimal"``.

        Returns
        -------
        str
            Full ASS file content.
        """
        from app.captions.styles import get_ass_style_block  # noqa: PLC0415

        style_block = get_ass_style_block(style)

        header = (
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
            f"{style_block}\n"
            "\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )

        dialogue_lines: List[str] = []
        for seg in self.segments:
            start_ts = self._seconds_to_ass_ts(seg.start)
            end_ts = self._seconds_to_ass_ts(seg.end)

            if seg.words and style == "dynamic":
                # Build karaoke-tagged text for word-level highlighting
                text_parts: List[str] = []
                for word in seg.words:
                    duration_cs = max(1, int(round((word.end - word.start) * 100)))
                    text_parts.append(f"{{\\k{duration_cs}}}{word.word.strip()}")
                text = " ".join(text_parts)
            else:
                text = seg.text.strip()

            style_name = {
                "dynamic": "Dynamic",
                "clean": "Clean",
                "minimal": "Minimal",
            }.get(style, "Default")

            dialogue_lines.append(
                f"Dialogue: 0,{start_ts},{end_ts},{style_name},,0,0,0,,{text}"
            )

        return header + "\n".join(dialogue_lines)
