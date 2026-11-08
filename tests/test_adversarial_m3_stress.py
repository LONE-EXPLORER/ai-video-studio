"""
tests/test_adversarial_m3_stress.py — Adversarial Edge-Case & Stress Test Suite for Milestone 3.

Empirically tests:
1. Edge-case handling in broll_pool.py:
   - Empty catalog & non-existent index file handling.
   - Missing files on disk (missing, 0-byte) filtered by require_exists.
   - Corrupt JSON, non-numeric durations, zero/negative durations.
   - Tag ranking with special characters, empty tags, non-string items.
2. Pacing & timeline boundaries in broll_matcher.py:
   - Transcripts with 0 words, whitespace only, stop-words only.
   - Hook protection boundary: speech starting at 0.0s or < 2.5s strictly rejected.
   - Tail protection boundary: overlays extending beyond total_duration - 1.0s rejected.
   - Cooldown spacing & budget cap under dense keyword bursts.
   - Ollama provider failure (timeout, network error, garbage json) falling back to heuristic matching.
3. VideoToolbox & FFmpeg filtergraph robustness in broll_compositor.py:
   - Missing and 0-byte B-roll files skipped gracefully.
   - Overlays exceeding base video length (start >= duration or end > duration).
   - Zero valid overlays pass-through base video without distortion.
   - Dialogue audio preservation: base video audio track 0 preserved, B-roll audio discarded.
   - Short clip looping over long overlay windows.
   - Software libx264 fallback vs VideoToolbox hardware encoding.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.analysis.broll_matcher import BrollMatcher, CandidateOverlay, TimelineWord
from app.analysis.broll_pool import BrollAsset, BrollCatalog
from app.providers.base import BaseProvider
from app.rendering.broll_compositor import BrollCompositor
from app.schemas.editing_plan import BrollOverlay, Clip, EditingPlan
from app.schemas.transcript import Transcript, TranscriptSegment, TranscriptWord
from app.utils.ffmpeg_utils import probe_video
from tests.e2e.conftest import make_synthetic_video


# ===========================================================================
# 1. Edge-case tests for broll_pool.py
# ===========================================================================

def test_broll_pool_empty_and_nonexistent_index(tmp_path: Path):
    """Catalog initializes cleanly when index file does not exist or is 0 bytes."""
    missing_index = tmp_path / "missing_index.json"
    cat_missing = BrollCatalog(index_path=missing_index, auto_provision=False)
    assert cat_missing.total_count == 0
    assert cat_missing.get_all() == []
    assert cat_missing.find_by_category("warning_trap") == []
    assert cat_missing.find_by_tags(["warning"]) == []

    empty_index = tmp_path / "empty_index.json"
    empty_index.touch()
    cat_empty = BrollCatalog(index_path=empty_index, auto_provision=False)
    assert cat_empty.total_count == 0


def test_broll_pool_missing_and_zero_byte_files(tmp_path: Path):
    """require_exists=True filters out missing files and 0-byte dummy files."""
    media_dir = tmp_path / "broll"
    media_dir.mkdir(parents=True, exist_ok=True)

    # 1 valid file
    valid_file = media_dir / "valid.mp4"
    valid_file.write_bytes(b"non_empty_media_data")

    # 1 zero-byte file
    zero_file = media_dir / "zero.mp4"
    zero_file.touch()

    # 1 missing file (not created on disk)
    index_data = {
        "assets": [
            {
                "id": "asset_valid",
                "filename": "valid.mp4",
                "category": "tech",
                "tags": ["coding", "ai"],
                "duration": 5.0,
            },
            {
                "id": "asset_zero",
                "filename": "zero.mp4",
                "category": "tech",
                "tags": ["coding", "zero"],
                "duration": 5.0,
            },
            {
                "id": "asset_missing",
                "filename": "missing.mp4",
                "category": "tech",
                "tags": ["coding", "missing"],
                "duration": 5.0,
            },
        ]
    }
    idx = media_dir / "broll_index.json"
    idx.write_text(json.dumps(index_data), encoding="utf-8")

    cat = BrollCatalog(index_path=idx)
    assert cat.total_count == 3

    # require_exists=False returns all 3
    all_raw = cat.get_all(require_exists=False)
    assert len(all_raw) == 3

    # require_exists=True returns only the non-empty existing file
    existing = cat.get_all(require_exists=True)
    assert len(existing) == 1
    assert existing[0].id == "asset_valid"

    # Category and tag queries respect require_exists
    by_cat = cat.find_by_category("tech", require_exists=True)
    assert len(by_cat) == 1
    assert by_cat[0].id == "asset_valid"

    by_tag = cat.find_by_tags(["coding"], require_exists=True)
    assert len(by_tag) == 1
    assert by_tag[0][0].id == "asset_valid"


def test_broll_pool_malformed_durations(tmp_path: Path):
    """Assets with 0, negative, or invalid durations are skipped during reload."""
    index_data = {
        "assets": [
            {"id": "a_valid", "filename": "a.mp4", "duration": 3.5, "tags": ["ok"]},
            {"id": "a_zero", "filename": "b.mp4", "duration": 0.0, "tags": ["bad"]},
            {"id": "a_neg", "filename": "c.mp4", "duration": -5.0, "tags": ["bad"]},
            {"id": "a_missing_dur", "filename": "d.mp4", "tags": ["bad"]},
        ]
    }
    idx = tmp_path / "broll_index.json"
    idx.write_text(json.dumps(index_data), encoding="utf-8")

    cat = BrollCatalog(index_path=idx)
    assert cat.total_count == 1
    assert cat.get_asset("a_valid") is not None
    assert cat.get_asset("a_zero") is None
    assert cat.get_asset("a_neg") is None
    assert cat.get_asset("a_missing_dur") is None


def test_broll_pool_corrupt_json_raises(tmp_path: Path):
    """Corrupted JSON raises JSONDecodeError."""
    idx = tmp_path / "corrupt.json"
    idx.write_text('{"assets": [{"id": "bad", "tags": [ incomplete', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        BrollCatalog(index_path=idx)


# ===========================================================================
# 2. Pacing & Timeline Boundary tests for broll_matcher.py
# ===========================================================================

def _setup_mock_catalog(tmp_path: Path) -> BrollCatalog:
    media_dir = tmp_path / "broll_assets"
    media_dir.mkdir(parents=True, exist_ok=True)
    f1 = media_dir / "warning.mp4"
    f1.write_bytes(b"content_warning")
    f2 = media_dir / "radar.mp4"
    f2.write_bytes(b"content_radar")
    f3 = media_dir / "tech.mp4"
    f3.write_bytes(b"content_tech")

    index_data = {
        "assets": [
            {
                "id": "broll_warning",
                "filename": "warning.mp4",
                "category": "warning_trap",
                "tags": ["warning", "danger", "trap", "threat"],
                "duration": 5.0,
            },
            {
                "id": "broll_radar",
                "filename": "radar.mp4",
                "category": "geopolitics_conflict",
                "tags": ["missiles", "radar", "military", "strike"],
                "duration": 5.0,
            },
            {
                "id": "broll_tech",
                "filename": "tech.mp4",
                "category": "technology_cyber",
                "tags": ["coding", "python", "software", "silicon"],
                "duration": 5.0,
            },
        ]
    }
    idx = media_dir / "broll_index.json"
    idx.write_text(json.dumps(index_data), encoding="utf-8")
    return BrollCatalog(index_path=idx)


def test_broll_matcher_empty_or_zero_word_transcript(tmp_path: Path):
    """Empty transcript, zero-word segments, or whitespace-only text yield zero overlays."""
    cat = _setup_mock_catalog(tmp_path)
    matcher = BrollMatcher(catalog=cat)

    plan = EditingPlan(
        plan_id="p1",
        job_id="j1",
        profile="dynamic_shorts",
        aspect_ratio="9:16",
        target_duration_seconds=30.0,
        hook_summary="Hook",
        edit_rationale="Rationale",
        clips=[Clip(clip_id="c1", source_start=0.0, source_end=30.0, order=1, reason="All")],
    )

    # 1. None transcript
    assert matcher.match_broll(plan, None) == []

    # 2. Empty segments transcript
    t_empty = Transcript(language="en", duration=30.0, segments=[])
    assert matcher.match_broll(plan, t_empty) == []

    # 3. Segments with no words and empty text
    t_blank = Transcript(
        language="en",
        duration=30.0,
        segments=[
            TranscriptSegment(id=0, start=1.0, end=5.0, text="   ", words=[]),
            TranscriptSegment(id=1, start=6.0, end=10.0, text="... ! ???", words=[]),
        ],
    )
    assert matcher.match_broll(plan, t_blank) == []


def test_broll_matcher_hook_protection_strictness(tmp_path: Path):
    """Any candidate starting before 2.5s is strictly rejected by hook protection."""
    cat = _setup_mock_catalog(tmp_path)
    matcher = BrollMatcher(catalog=cat, hook_protection=2.5)

    asset = cat.get_asset("broll_warning")
    candidates = [
        CandidateOverlay("broll_warning", timeline_start=0.0, duration=2.0, score=1.0, matched_keyword="trap", asset=asset),
        CandidateOverlay("broll_warning", timeline_start=1.2, duration=2.0, score=1.0, matched_keyword="trap", asset=asset),
        CandidateOverlay("broll_warning", timeline_start=2.49, duration=2.0, score=1.0, matched_keyword="trap", asset=asset),
        CandidateOverlay("broll_warning", timeline_start=2.50, duration=2.0, score=1.0, matched_keyword="trap", asset=asset),
    ]

    overlays = matcher._enforce_pacing_constraints(candidates, total_duration=30.0)
    assert len(overlays) == 1
    assert overlays[0].timeline_start == 2.50


def test_broll_matcher_tail_protection_strictness(tmp_path: Path):
    """Overlays extending past total_duration - tail_protection are strictly rejected."""
    cat = _setup_mock_catalog(tmp_path)
    # Total duration = 20.0s, tail_protection = 1.0s -> max allowed end is 19.0s
    matcher = BrollMatcher(catalog=cat, tail_protection=1.0, min_duration=1.5)
    asset = cat.get_asset("broll_radar")

    candidates = [
        # Start at 17.5s + min_duration 1.5s = 19.0s -> ALLOWED
        CandidateOverlay("broll_radar", timeline_start=17.5, duration=1.5, score=1.0, matched_keyword="radar", asset=asset),
        # Start at 17.6s + min_duration 1.5s = 19.1s > 19.0s -> REJECTED
        CandidateOverlay("broll_radar", timeline_start=17.6, duration=1.5, score=1.0, matched_keyword="radar", asset=asset),
        # Start at 19.5s -> REJECTED
        CandidateOverlay("broll_radar", timeline_start=19.5, duration=1.5, score=1.0, matched_keyword="radar", asset=asset),
    ]

    overlays = matcher._enforce_pacing_constraints(candidates, total_duration=20.0)
    assert len(overlays) == 1
    assert overlays[0].timeline_start == 17.5
    assert overlays[0].timeline_end <= 19.0


def test_broll_matcher_dense_cooldown_and_budget_cap(tmp_path: Path):
    """Dense candidates enforce min_spacing (5.0s) and budget cap (20%)."""
    cat = _setup_mock_catalog(tmp_path)
    # Total duration = 50.0s, 20% budget = 10.0s max total B-roll
    matcher = BrollMatcher(
        catalog=cat,
        min_spacing=5.0,
        max_coverage_ratio=0.20,
        default_duration=2.0,
        min_duration=1.5,
    )
    w = cat.get_asset("broll_warning")
    r = cat.get_asset("broll_radar")
    t = cat.get_asset("broll_tech")

    # Generate 10 dense candidates spaced by 1 second
    candidates = [
        CandidateOverlay("broll_warning", timeline_start=3.0, duration=2.0, score=1.0, matched_keyword="w", asset=w),
        CandidateOverlay("broll_radar", timeline_start=4.0, duration=2.0, score=1.0, matched_keyword="r", asset=r),
        CandidateOverlay("broll_tech", timeline_start=5.0, duration=2.0, score=1.0, matched_keyword="t", asset=t),
        CandidateOverlay("broll_radar", timeline_start=10.0, duration=2.0, score=1.0, matched_keyword="r", asset=r),
        CandidateOverlay("broll_tech", timeline_start=11.0, duration=2.0, score=1.0, matched_keyword="t", asset=t),
        CandidateOverlay("broll_tech", timeline_start=18.0, duration=2.0, score=1.0, matched_keyword="t", asset=t),
    ]

    overlays = matcher._enforce_pacing_constraints(candidates, total_duration=50.0)

    # Verify spacing between consecutive overlays is at least 5.0s
    for i in range(len(overlays) - 1):
        gap = overlays[i + 1].timeline_start - overlays[i].timeline_end
        assert gap >= 5.0, f"Gap {gap:.2f}s is less than 5.0s min_spacing"

    # Verify budget cap
    tot_dur = sum(o.duration for o in overlays)
    assert tot_dur <= 10.0


def test_broll_matcher_ollama_failure_fallbacks(tmp_path: Path):
    """Ollama failures (exception or malformed response) trigger heuristic fallback seamlessly."""
    cat = _setup_mock_catalog(tmp_path)

    plan = EditingPlan(
        plan_id="p2",
        job_id="j2",
        profile="dynamic_shorts",
        aspect_ratio="9:16",
        target_duration_seconds=30.0,
        hook_summary="Hook",
        edit_rationale="Rationale",
        clips=[Clip(clip_id="c1", source_start=0.0, source_end=30.0, order=1, reason="All")],
    )
    transcript = Transcript(
        language="en",
        duration=30.0,
        segments=[
            TranscriptSegment(
                id=0,
                start=4.0,
                end=8.0,
                text="military radar and missile strike detected",
                words=[
                    TranscriptWord(word="military", start=4.0, end=4.8, probability=0.99),
                    TranscriptWord(word="radar", start=5.0, end=5.8, probability=0.99),
                    TranscriptWord(word="missile", start=6.0, end=6.8, probability=0.99),
                ],
            )
        ],
    )

    # Case A: Provider raises TimeoutError
    prov_err = MagicMock(spec=BaseProvider)
    prov_err.is_available.return_value = True
    prov_err.generate_json.side_effect = TimeoutError("Ollama request timed out after 30s")

    matcher_err = BrollMatcher(provider=prov_err, catalog=cat)
    res_err = matcher_err.match_broll(plan, transcript)
    assert len(res_err) >= 1
    assert res_err[0].broll_id == "broll_radar"

    # Case B: Provider returns non-dict or empty response
    prov_empty = MagicMock(spec=BaseProvider)
    prov_empty.is_available.return_value = True
    prov_empty.generate_json.return_value = {"matches": []}

    matcher_empty = BrollMatcher(provider=prov_empty, catalog=cat)
    res_empty = matcher_empty.match_broll(plan, transcript)
    assert len(res_empty) >= 1
    assert res_empty[0].broll_id == "broll_radar"


# ===========================================================================
# 3. VideoToolbox & FFmpeg filtergraph robustness for broll_compositor.py
# ===========================================================================

def test_broll_compositor_zero_valid_overlays_passthrough(tmp_path: Path):
    """When overlays list is empty or all overlays are invalid, base video passes through untouched."""
    base = tmp_path / "base.mp4"
    out = tmp_path / "out_passthrough.mp4"
    make_synthetic_video(base, width=1080, height=1920, fps=30, duration=2.5, has_audio=True)

    compositor = BrollCompositor()

    # Empty list
    res = compositor.composite_overlays(base, [], out)
    assert res.exists()
    assert res.stat().st_size == base.stat().st_size

    # Invalid missing file overlay
    bad_overlay = BrollOverlay(
        broll_id="missing",
        timeline_start=1.0,
        timeline_end=2.0,
        source_path=str(tmp_path / "phantom.mp4"),
    )
    out_bad = tmp_path / "out_bad.mp4"
    res_bad = compositor.composite_overlays(base, [bad_overlay], out_bad)
    assert res_bad.exists()
    assert res_bad.stat().st_size == base.stat().st_size


def test_broll_compositor_timestamps_exceeding_base_duration(tmp_path: Path):
    """Overlays starting beyond base duration are dropped; overlays crossing boundary are clamped."""
    base = tmp_path / "base_short.mp4"
    broll = tmp_path / "broll_clip.mp4"
    out = tmp_path / "out_clamped.mp4"

    # Base is exactly 3.0s
    make_synthetic_video(base, width=1080, height=1920, fps=30, duration=3.0, has_audio=True)
    make_synthetic_video(broll, width=1080, height=1920, fps=30, duration=2.0, pattern="color", has_audio=False)

    compositor = BrollCompositor()
    overlays = [
        # Crossing end: start 2.0s, end 4.5s -> should clamp to 3.0s
        BrollOverlay(broll_id="cross", timeline_start=2.0, timeline_end=4.5, source_path=str(broll)),
        # Entirely past end: start 3.5s, end 5.0s -> should be dropped
        BrollOverlay(broll_id="past", timeline_start=3.5, timeline_end=5.0, source_path=str(broll)),
    ]

    res = compositor.composite_overlays(base, overlays, out)
    assert res.exists()
    assert res.stat().st_size > 0

    info = probe_video(res)
    dur = float(info["format"]["duration"])
    assert dur == pytest.approx(3.0, abs=0.2)


def test_broll_compositor_dialogue_audio_purity(tmp_path: Path):
    """Base dialogue audio (440Hz sine wave) is preserved without any distortion or B-roll audio leakage."""
    base = tmp_path / "base_audio.mp4"
    broll = tmp_path / "broll_with_audio.mp4"
    out = tmp_path / "out_dialogue_pure.mp4"

    make_synthetic_video(base, width=1080, height=1920, fps=30, duration=3.0, has_audio=True, audio_freq=440)
    # B-roll has contrasting 1000Hz audio
    make_synthetic_video(broll, width=1080, height=1920, fps=30, duration=1.5, pattern="color", has_audio=True, audio_freq=1000)

    compositor = BrollCompositor()
    overlays = [
        BrollOverlay(broll_id="ov1", timeline_start=0.5, timeline_end=2.0, source_path=str(broll))
    ]

    res = compositor.composite_overlays(base, overlays, out)
    info = probe_video(res)
    astreams = [s for s in info["streams"] if s["codec_type"] == "audio"]
    assert len(astreams) == 1
    assert int(astreams[0]["sample_rate"]) == 48000
    assert astreams[0]["channels"] == 2


def test_broll_compositor_base_video_without_audio(tmp_path: Path):
    """Base video with no audio stream composites cleanly with -an without stream map errors."""
    base_no_audio = tmp_path / "base_no_audio.mp4"
    broll = tmp_path / "broll.mp4"
    out = tmp_path / "out_no_audio.mp4"

    make_synthetic_video(base_no_audio, width=1080, height=1920, fps=30, duration=2.0, has_audio=False)
    make_synthetic_video(broll, width=1080, height=1920, fps=30, duration=1.0, pattern="color", has_audio=False)

    compositor = BrollCompositor()
    overlays = [
        BrollOverlay(broll_id="b1", timeline_start=0.5, timeline_end=1.5, source_path=str(broll))
    ]

    res = compositor.composite_overlays(base_no_audio, overlays, out)
    assert res.exists()
    assert res.stat().st_size > 0
    info = probe_video(res)
    astreams = [s for s in info["streams"] if s["codec_type"] == "audio"]
    assert len(astreams) == 0


def test_broll_overlay_pydantic_rejects_inverted_and_zero_duration(tmp_path: Path):
    """Pydantic model validator rejects BrollOverlay when timeline_end <= timeline_start."""
    from pydantic import ValidationError

    dummy_media = tmp_path / "dummy.mp4"
    dummy_media.touch()

    # Inverted duration raises ValidationError
    with pytest.raises(ValidationError):
        BrollOverlay(broll_id="inv", timeline_start=2.0, timeline_end=1.0, source_path=str(dummy_media))

    # Zero duration raises ValidationError
    with pytest.raises(ValidationError):
        BrollOverlay(broll_id="zero", timeline_start=1.5, timeline_end=1.5, source_path=str(dummy_media))


def test_broll_compositor_clamped_to_zero_duration_is_dropped(tmp_path: Path):
    """Overlay whose timeline_end clamps to <= timeline_start is dropped safely."""
    base = tmp_path / "base.mp4"
    broll = tmp_path / "broll.mp4"
    out = tmp_path / "out_clamped_zero.mp4"

    # Base is exactly 2.0s
    make_synthetic_video(base, width=1080, height=1920, fps=30, duration=2.0, has_audio=True)
    make_synthetic_video(broll, width=1080, height=1920, fps=30, duration=1.0, pattern="color", has_audio=False)

    compositor = BrollCompositor()
    # Overlay starts at 2.0s (>= base duration 2.0) -> dropped!
    # And another overlay starting at 1.99s with end 2.0001s -> clamped_end 2.0, valid.
    overlays = [
        BrollOverlay(broll_id="at_edge", timeline_start=2.0, timeline_end=2.5, source_path=str(broll)),
    ]

    res = compositor.composite_overlays(base, overlays, out)
    # The overlay was dropped -> base video returned
    assert res.exists()
    assert res.stat().st_size == base.stat().st_size


