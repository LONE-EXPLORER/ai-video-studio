"""
tests/test_sidechain_ducking.py — Tests for SidechainDucker dynamic music ducking.
"""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.editing.sidechain_ducker import SidechainDucker, DuckingParams


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_ducker():
    return SidechainDucker()

@pytest.fixture
def custom_ducker():
    return SidechainDucker(DuckingParams(
        attack_ms=10.0,
        release_ms=300.0,
        threshold_db=-20.0,
        ratio=6.0,
        music_pre_vol=0.4,
    ))

@pytest.fixture
def fake_video(tmp_path):
    p = tmp_path / "dialogue.mp4"
    p.write_bytes(b"fake_video" * 500)
    return p

@pytest.fixture
def fake_music(tmp_path):
    p = tmp_path / "music.wav"
    p.write_bytes(b"fake_music" * 500)
    return p


# ---------------------------------------------------------------------------
# DuckingParams tests
# ---------------------------------------------------------------------------

class TestDuckingParams:

    def test_default_params(self):
        p = DuckingParams()
        assert p.attack_ms == 5.0
        assert p.release_ms == 250.0
        assert p.threshold_db == -24.0
        assert p.ratio == 8.0
        assert 0.0 < p.music_pre_vol < 1.0

    def test_custom_params(self):
        p = DuckingParams(attack_ms=15.0, release_ms=400.0, ratio=12.0)
        assert p.attack_ms == 15.0
        assert p.release_ms == 400.0
        assert p.ratio == 12.0

    def test_threshold_to_linear_conversion(self):
        """Verify dB to linear threshold logic used in filter_complex."""
        import math
        p = DuckingParams(threshold_db=-24.0)
        linear = 10 ** (p.threshold_db / 20.0)
        assert abs(linear - 0.0631) < 0.001, "−24dB should be ~0.0631 linear"


# ---------------------------------------------------------------------------
# SidechainDucker: FFmpeg command generation tests
# ---------------------------------------------------------------------------

class TestSidechainDuckerFilterComplex:

    def test_success_path_calls_ffmpeg(self, default_ducker, fake_video, fake_music, tmp_path):
        out = tmp_path / "ducked.mp4"
        with patch("app.editing.sidechain_ducker.subprocess.run") as mock_run:
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_run.return_value = mock_proc
            out.write_bytes(b"fake_ducked" * 500)
            result = default_ducker.duck_and_mix(fake_video, fake_music, out, music_volume=0.35)
        assert mock_run.called
        cmd = mock_run.call_args[0][0]
        assert "sidechaincompress" in " ".join(cmd), "Must use sidechain compress filter"
        assert "loudnorm" in " ".join(cmd), "Must include loudnorm for broadcast levels"

    def test_fallback_on_ffmpeg_failure(self, default_ducker, fake_video, fake_music, tmp_path):
        out = tmp_path / "ducked.mp4"
        with patch("app.editing.sidechain_ducker.subprocess.run") as mock_run:
            # First call (sidechain) fails; second call (static fallback) succeeds
            mock_fail = MagicMock()
            mock_fail.returncode = 1
            mock_fail.stderr = "sidechain error"
            mock_ok = MagicMock()
            mock_ok.returncode = 0
            mock_run.side_effect = [mock_fail, mock_ok]
            out.write_bytes(b"fallback_output" * 100)
            result = default_ducker.duck_and_mix(fake_video, fake_music, out, music_volume=0.35)
        assert mock_run.call_count >= 2, "Should retry with static fallback"

    def test_fallback_uses_volume_filter(self, default_ducker, fake_video, fake_music, tmp_path):
        out = tmp_path / "ducked.mp4"
        with patch("app.editing.sidechain_ducker.subprocess.run") as mock_run:
            mock_fail = MagicMock()
            mock_fail.returncode = 1
            mock_fail.stderr = ""
            mock_ok = MagicMock()
            mock_ok.returncode = 0
            mock_run.side_effect = [mock_fail, mock_ok]
            out.write_bytes(b"x" * 100)
            default_ducker.duck_and_mix(fake_video, fake_music, out, music_volume=0.35)
        # Second call should contain 'volume' instead of 'sidechaincompress'
        second_cmd = " ".join(mock_run.call_args_list[1][0][0])
        assert "volume" in second_cmd

    def test_timeout_falls_back_to_copy(self, default_ducker, fake_video, fake_music, tmp_path):
        import subprocess
        out = tmp_path / "ducked.mp4"
        with patch("app.editing.sidechain_ducker.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=180)
            # Second call (static fallback) should also raise, triggering file copy
            result = default_ducker.duck_and_mix(fake_video, fake_music, out, music_volume=0.35)
        assert out.exists()

    def test_attack_and_release_appear_in_filter(self, custom_ducker, fake_video, fake_music, tmp_path):
        out = tmp_path / "ducked.mp4"
        with patch("app.editing.sidechain_ducker.subprocess.run") as mock_run:
            mock_ok = MagicMock()
            mock_ok.returncode = 0
            mock_run.return_value = mock_ok
            out.write_bytes(b"y" * 100)
            custom_ducker.duck_and_mix(fake_video, fake_music, out)
        cmd_str = " ".join(mock_run.call_args[0][0])
        assert "attack=10" in cmd_str or "attack=10.0" in cmd_str
        assert "release=300" in cmd_str or "release=300.0" in cmd_str


# ---------------------------------------------------------------------------
# Integration stub: static fallback
# ---------------------------------------------------------------------------

class TestStaticFallback:

    def test_static_fallback_includes_volume(self, default_ducker, fake_video, fake_music, tmp_path):
        out = tmp_path / "static.mp4"
        with patch("app.editing.sidechain_ducker.subprocess.run") as mock_run:
            mock_ok = MagicMock()
            mock_ok.returncode = 0
            mock_run.return_value = mock_ok
            out.write_bytes(b"z" * 100)
            default_ducker._static_fallback(fake_video, fake_music, out, music_volume=0.3)
        cmd_str = " ".join(mock_run.call_args[0][0])
        assert "volume=0.3" in cmd_str or "0.300" in cmd_str

    def test_static_fallback_shortest_flag(self, default_ducker, fake_video, fake_music, tmp_path):
        out = tmp_path / "static.mp4"
        with patch("app.editing.sidechain_ducker.subprocess.run") as mock_run:
            mock_ok = MagicMock()
            mock_ok.returncode = 0
            mock_run.return_value = mock_ok
            out.write_bytes(b"z" * 100)
            default_ducker._static_fallback(fake_video, fake_music, out, music_volume=0.35)
        cmd = mock_run.call_args[0][0]
        assert "-shortest" in cmd, "Must trim output to shortest input (video or music)"
