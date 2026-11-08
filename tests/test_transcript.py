"""
tests/test_transcript.py — Tests for transcript serialization and subtitle conversion.
"""

from __future__ import annotations

from pathlib import Path
from app.schemas.transcript import Transcript
from app.transcription.srt_generator import generate_ass, generate_srt


def test_generate_srt(sample_transcript: Transcript, tmp_path: Path):
    srt_out = tmp_path / "test.srt"
    generate_srt(sample_transcript, srt_out)
    assert srt_out.exists()
    content = srt_out.read_text(encoding="utf-8")
    assert "00:00:00,500 --> 00:00:04,000" in content
    assert "Welcome to this video on AI video editing." in content


def test_generate_ass(sample_transcript: Transcript, tmp_path: Path):
    ass_out = tmp_path / "test.ass"
    generate_ass(sample_transcript, "dynamic", ass_out)
    assert ass_out.exists()
    content = ass_out.read_text(encoding="utf-8")
    assert "[Script Info]" in content
    assert "[V4+ Styles]" in content
    assert "Dialogue:" in content
