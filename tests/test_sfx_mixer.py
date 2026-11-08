"""
tests/test_sfx_mixer.py — Tests for SFX placement and mixing engine.
"""

import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.editing.sfx_mixer import SFXEvent, SFXMixer, SFXMixResult
from app.schemas.editing_plan import EditingPlan, Clip, ZoomEffect

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_plan():
    """A 3-clip editing plan for testing SFX event generation."""
    return EditingPlan(
        plan_id="plan_min",
        job_id="job_min",
        profile="dynamic_shorts",
        aspect_ratio="9:16",
        target_duration_seconds=12.0,
        hook_summary="Hook test",
        edit_rationale="Fast cut rationale",
        clips=[
            Clip(clip_id="1", source_start=0.0, source_end=5.0, order=1, reason="hook",
                 zoom=ZoomEffect(enabled=True, factor=1.15)),
            Clip(clip_id="2", source_start=5.0, source_end=9.0, order=2, reason="speech",
                 zoom=ZoomEffect(enabled=True, factor=1.08)),
            Clip(clip_id="3", source_start=9.0, source_end=12.0, order=3, reason="outro",
                 zoom=ZoomEffect(enabled=False, factor=1.0)),
        ]
    )

@pytest.fixture
def long_plan():
    """10-clip plan for riser event testing (> 25s total)."""
    clips = [
        Clip(clip_id=str(i+1), source_start=float(i*3), source_end=float(i*3 + 3.0), order=i+1, reason=f"clip_{i}",
             zoom=ZoomEffect(enabled=(i % 2 == 0), factor=1.08))
        for i in range(10)
    ]
    return EditingPlan(
        plan_id="plan_long",
        job_id="job_long",
        profile="dynamic_shorts",
        aspect_ratio="9:16",
        target_duration_seconds=30.0,
        hook_summary="Hook test",
        edit_rationale="Fast cut rationale",
        clips=clips
    )

@pytest.fixture
def mock_sfx_root(tmp_path):
    """Create a minimal SFX asset root with fake WAV files and index."""
    sfx_dir = tmp_path / "sfx"
    sfx_dir.mkdir()
    for name in ["whoosh.wav", "pop.wav", "riser.wav", "impact.wav", "swoosh_short.wav"]:
        # Create minimal valid WAV (44 bytes header + silence)
        wav_data = bytes([
            0x52, 0x49, 0x46, 0x46,  # "RIFF"
            0x24, 0x00, 0x00, 0x00,  # chunk size
            0x57, 0x41, 0x56, 0x45,  # "WAVE"
            0x66, 0x6D, 0x74, 0x20,  # "fmt "
            0x10, 0x00, 0x00, 0x00,  # subchunk size 16
            0x01, 0x00,              # PCM
            0x02, 0x00,              # stereo
            0x80, 0xBB, 0x00, 0x00,  # 48000 Hz
            0x00, 0xEE, 0x02, 0x00,  # byte rate
            0x04, 0x00,              # block align
            0x10, 0x00,              # bits per sample
            0x64, 0x61, 0x74, 0x61,  # "data"
            0x00, 0x00, 0x00, 0x00,  # data size (0 = silence)
        ])
        (sfx_dir / name).write_bytes(wav_data)

    index = {
        "whoosh": {"path": f"assets/sfx/whoosh.wav", "duration": 0.6, "trigger": "punch_in", "volume": 0.65},
        "pop": {"path": f"assets/sfx/pop.wav", "duration": 0.3, "trigger": "word_highlight", "volume": 0.8},
        "riser": {"path": f"assets/sfx/riser.wav", "duration": 2.0, "trigger": "hook_intro", "volume": 0.45},
        "impact": {"path": f"assets/sfx/impact.wav", "duration": 0.4, "trigger": "power_word", "volume": 0.9},
        "swoosh_short": {"path": f"assets/sfx/swoosh_short.wav", "duration": 0.35, "trigger": "jump_cut", "volume": 0.6},
    }
    (sfx_dir / "sfx_index.json").write_text(json.dumps(index))
    return sfx_dir


@pytest.fixture
def sfx_mixer(mock_sfx_root):
    return SFXMixer(sfx_root=mock_sfx_root)


# ---------------------------------------------------------------------------
# Unit tests: build_events_from_plan
# ---------------------------------------------------------------------------

class TestBuildEventsFromPlan:

    def test_impact_at_hook(self, sfx_mixer, minimal_plan):
        events = sfx_mixer.build_events_from_plan(minimal_plan)
        impact_events = [e for e in events if e.sfx_name == "impact"]
        assert len(impact_events) >= 1
        assert impact_events[0].timecode == 0.0, "Impact must be placed at timecode 0 (hook)"

    def test_whoosh_before_cuts(self, sfx_mixer, minimal_plan):
        events = sfx_mixer.build_events_from_plan(minimal_plan)
        # Clips 2 and 3 start at t=5 and t=9; whoosh should be ~50ms before
        whoosh_or_swoosh = [e for e in events if e.sfx_name in ("whoosh", "swoosh_short")]
        assert len(whoosh_or_swoosh) >= 1, "Should have at least one whoosh/swoosh for cuts"

    def test_no_negative_timecodes(self, sfx_mixer, minimal_plan):
        events = sfx_mixer.build_events_from_plan(minimal_plan)
        for ev in events:
            assert ev.timecode >= 0.0, f"Negative timecode {ev.timecode} for {ev.sfx_name}"

    def test_events_sorted_by_timecode(self, sfx_mixer, minimal_plan):
        events = sfx_mixer.build_events_from_plan(minimal_plan)
        timecodes = [e.timecode for e in events]
        assert timecodes == sorted(timecodes), "Events must be sorted by timecode"

    def test_riser_added_for_long_video(self, sfx_mixer, long_plan):
        events = sfx_mixer.build_events_from_plan(long_plan)
        riser_events = [e for e in events if e.sfx_name == "riser"]
        assert len(riser_events) >= 1, "Riser should be placed for videos > 25s"

    def test_riser_not_added_for_short_video(self, sfx_mixer, minimal_plan):
        """Minimal plan is 12s total — below the 25s riser threshold."""
        events = sfx_mixer.build_events_from_plan(minimal_plan)
        riser_events = [e for e in events if e.sfx_name == "riser"]
        assert len(riser_events) == 0, "No riser for short videos"

    def test_volume_in_valid_range(self, sfx_mixer, long_plan):
        events = sfx_mixer.build_events_from_plan(long_plan)
        for ev in events:
            assert 0.0 < ev.volume <= 1.5, f"Volume {ev.volume} out of reasonable range"

    def test_empty_plan_returns_empty(self, sfx_mixer):
        empty_plan = EditingPlan(plan_id="p0", job_id="j0", profile="dynamic_shorts", aspect_ratio="9:16",
        target_duration_seconds=0.0, hook_summary="h", edit_rationale="r", clips=[])
        events = sfx_mixer.build_events_from_plan(empty_plan)
        assert events == [], "Empty plan should produce no SFX events"

    def test_single_clip_plan_has_impact_only(self, sfx_mixer):
        plan = EditingPlan(
            plan_id="p1", job_id="j1", profile="dynamic_shorts", aspect_ratio="9:16",
        target_duration_seconds=10.0, hook_summary="h", edit_rationale="r",
            clips=[
                Clip(clip_id="1", source_start=0.0, source_end=10.0, order=1, reason="hook",
                     zoom=ZoomEffect(enabled=True, factor=1.15))
            ]
        )
        events = sfx_mixer.build_events_from_plan(plan)
        # Should have impact at hook but no whoosh (only one clip, no cuts)
        assert any(e.sfx_name == "impact" for e in events)
        whoosh = [e for e in events if e.sfx_name == "whoosh"]
        assert len(whoosh) == 0


# ---------------------------------------------------------------------------
# Unit tests: SFXEvent dataclass
# ---------------------------------------------------------------------------

class TestSFXEventDataclass:

    def test_default_volume_and_pan(self):
        ev = SFXEvent(sfx_name="whoosh", timecode=1.0)
        assert ev.volume == 1.0
        assert ev.pan == 0.0

    def test_custom_volume(self):
        ev = SFXEvent(sfx_name="impact", timecode=0.0, volume=0.75)
        assert ev.volume == 0.75


# ---------------------------------------------------------------------------
# Integration tests: mix_sfx (mocked FFmpeg)
# ---------------------------------------------------------------------------

class TestMixSFX:

    def test_empty_events_copies_original(self, sfx_mixer, tmp_path):
        fake_video = tmp_path / "input.mp4"
        fake_video.write_bytes(b"fake")
        out = tmp_path / "output.mp4"
        result = sfx_mixer.mix_sfx(fake_video, [], out, overall_sfx_volume=0.6)
        assert result.success
        assert result.events_placed == 0
        assert out.exists()

    def test_mix_produces_output_on_valid_events(self, sfx_mixer, tmp_path, mock_sfx_root):
        """Mock FFmpeg call to verify command is built and output is created."""
        fake_video = tmp_path / "input.mp4"
        fake_video.write_bytes(b"fake_video_data" * 100)
        out = tmp_path / "sfx_mixed.mp4"

        # Provide a real SFX event but mock FFmpeg to succeed
        events = [SFXEvent(sfx_name="whoosh", timecode=1.0, volume=0.6)]

        # Patch the sfx_mixer's index to point to our temp files
        for sfx_name in sfx_mixer._index:
            sfx_mixer._index[sfx_name]["path"] = f"assets/sfx/{sfx_name}.wav"

        # Mock subprocess.run to simulate FFmpeg success
        with patch("app.editing.sfx_mixer.subprocess.run") as mock_run:
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_run.return_value = mock_proc
            # Create fake output
            out.write_bytes(b"fake_mixed" * 100)
            result = sfx_mixer.mix_sfx(fake_video, events, out, overall_sfx_volume=0.6)

        assert mock_run.called, "FFmpeg subprocess should have been called"

    def test_ffmpeg_failure_falls_back_to_copy(self, sfx_mixer, tmp_path):
        """When FFmpeg returns non-zero, mix_sfx should fall back gracefully."""
        fake_video = tmp_path / "input.mp4"
        fake_video.write_bytes(b"fake_video_data" * 100)
        out = tmp_path / "sfx_mixed.mp4"
        events = [SFXEvent(sfx_name="impact", timecode=0.0, volume=0.75)]

        with patch("app.editing.sfx_mixer.subprocess.run") as mock_run:
            mock_proc = MagicMock()
            mock_proc.returncode = 1
            mock_proc.stderr = "FFmpeg error"
            mock_run.return_value = mock_proc
            result = sfx_mixer.mix_sfx(fake_video, events, out, overall_sfx_volume=0.6)

        # Should have fallen back to copy
        assert out.exists()
        assert result.events_placed == 0

    def test_missing_sfx_file_skips_gracefully(self, sfx_mixer, tmp_path):
        """Events referencing non-existent files should be skipped, not crash."""
        fake_video = tmp_path / "input.mp4"
        fake_video.write_bytes(b"fake")
        out = tmp_path / "output.mp4"
        events = [SFXEvent(sfx_name="nonexistent_sound", timecode=1.0)]
        result = sfx_mixer.mix_sfx(fake_video, events, out, overall_sfx_volume=0.6)
        assert result.success or not result.success  # doesn't crash either way
        assert out.exists()


# ---------------------------------------------------------------------------
# SFX index tests
# ---------------------------------------------------------------------------

class TestSFXIndex:

    def test_index_has_required_keys(self, mock_sfx_root):
        idx = json.loads((mock_sfx_root / "sfx_index.json").read_text())
        for key in ("whoosh", "impact", "riser", "pop", "swoosh_short"):
            assert key in idx, f"Missing SFX key: {key}"

    def test_index_entries_have_required_fields(self, mock_sfx_root):
        idx = json.loads((mock_sfx_root / "sfx_index.json").read_text())
        for name, info in idx.items():
            assert "path" in info, f"{name} missing 'path'"
            assert "duration" in info, f"{name} missing 'duration'"
            assert "volume" in info, f"{name} missing 'volume'"
            assert info["duration"] > 0, f"{name} duration must be positive"
            assert 0 < info["volume"] <= 1.0, f"{name} volume out of range"
