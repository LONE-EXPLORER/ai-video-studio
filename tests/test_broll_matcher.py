"""
tests/test_broll_matcher.py — Unit tests for Semantic Transcript B-Roll Matcher.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

from app.analysis.broll_matcher import BrollMatcher, CandidateOverlay, TimelineWord
from app.analysis.broll_pool import BrollAsset, BrollCatalog
from app.providers.base import BaseProvider
from app.schemas.editing_plan import BrollOverlay, Clip, EditingPlan
from app.schemas.transcript import Transcript, TranscriptSegment, TranscriptWord


def _make_sample_catalog(tmp_path: Path) -> BrollCatalog:
    """Create a temporary catalog with mock test media files."""
    media_dir = tmp_path / "broll"
    media_dir.mkdir(parents=True, exist_ok=True)

    tech_file = media_dir / "tech.mp4"
    tech_file.write_bytes(b"dummy_mp4_tech")
    money_file = media_dir / "money.mp4"
    money_file.write_bytes(b"dummy_mp4_money")
    geo_file = media_dir / "geo.mp4"
    geo_file.write_bytes(b"dummy_mp4_geo")

    index_data = {
        "assets": [
            {
                "id": "broll_tech",
                "filename": "tech.mp4",
                "category": "technology_cyber",
                "tags": ["coding", "python", "developer", "software", "apple silicon"],
                "duration": 5.0,
                "description": "Tech code matrix",
            },
            {
                "id": "broll_money",
                "filename": "money.mp4",
                "category": "finance_money",
                "tags": ["finance", "money", "growth", "stock", "crypto", "wealth"],
                "duration": 5.0,
                "description": "Financial stock chart",
            },
            {
                "id": "broll_geo",
                "filename": "geo.mp4",
                "category": "geopolitics_conflict",
                "tags": ["military", "missiles", "radar", "war", "conflict", "iran"],
                "duration": 5.0,
                "description": "Tactical radar grid",
            },
        ]
    }
    idx_file = media_dir / "broll_index.json"
    idx_file.write_text(json.dumps(index_data), encoding="utf-8")
    return BrollCatalog(index_path=idx_file)


def test_timeline_word_mapping():
    """Verify source transcript words are mapped to timeline coordinates across cuts."""
    matcher = BrollMatcher()
    clips = [
        Clip(clip_id="c001", source_start=10.0, source_end=15.0, order=1, reason="Beat 1"),  # 5.0s dur
        Clip(clip_id="c002", source_start=25.0, source_end=35.0, order=2, reason="Beat 2"),  # 10.0s dur
    ]
    # Total duration = 15.0s
    # Clip 1 timeline = [0.0, 5.0]
    # Clip 2 timeline = [5.0, 15.0]

    transcript = Transcript(
        language="en",
        duration=40.0,
        segments=[
            TranscriptSegment(
                id=0,
                start=11.0,
                end=13.0,
                text="coding in Python",
                words=[
                    TranscriptWord(word="coding", start=11.0, end=12.0, probability=0.99),
                    TranscriptWord(word="Python", start=12.2, end=12.8, probability=0.99),
                ],
            ),
            TranscriptSegment(
                id=1,
                start=18.0,  # Dropped silence/cut segment outside clips
                end=22.0,
                text="cut silence words",
                words=[
                    TranscriptWord(word="cut", start=18.0, end=19.0, probability=0.99),
                ],
            ),
            TranscriptSegment(
                id=2,
                start=27.0,
                end=29.0,
                text="stocks",
                words=[
                    TranscriptWord(word="stocks", start=27.0, end=28.5, probability=0.99),
                ],
            ),
        ],
    )

    timeline_words = matcher._map_transcript_to_timeline(clips, transcript)
    clean_words = [w.clean_word for w in timeline_words]
    assert "coding" in clean_words
    assert "python" in clean_words
    assert "stocks" in clean_words
    assert "cut" not in clean_words  # Cut interval excluded!

    coding_tw = next(w for w in timeline_words if w.clean_word == "coding")
    # source 11.0 - clip_start 10.0 + offset 0.0 = 1.0s
    assert pytest.approx(coding_tw.timeline_start, abs=0.01) == 1.0

    stocks_tw = next(w for w in timeline_words if w.clean_word == "stocks")
    # source 27.0 - clip_start 25.0 + offset 5.0 = 7.0s
    assert pytest.approx(stocks_tw.timeline_start, abs=0.01) == 7.0


def test_heuristic_exact_and_fuzzy_matching(tmp_path: Path):
    """Verify exact tag matches and fuzzy n-gram matches trigger appropriate candidates."""
    catalog = _make_sample_catalog(tmp_path)
    matcher = BrollMatcher(catalog=catalog)

    timeline_words = [
        TimelineWord(word="we", clean_word="we", timeline_start=1.0, timeline_end=1.2),
        TimelineWord(word="built", clean_word="built", timeline_start=1.3, timeline_end=1.5),
        TimelineWord(word="software", clean_word="software", timeline_start=3.5, timeline_end=4.0),
        TimelineWord(word="for", clean_word="for", timeline_start=4.1, timeline_end=4.3),
        TimelineWord(word="financial", clean_word="financial", timeline_start=10.0, timeline_end=10.5),
    ]

    candidates = matcher._match_with_heuristics(timeline_words, total_duration=25.0)
    matched_ids = {c.broll_id for c in candidates}
    assert "broll_tech" in matched_ids  # exact "software"
    assert "broll_money" in matched_ids  # fuzzy/substring "financial" -> "finance"


def test_hook_protection_rule(tmp_path: Path):
    """Verify B-roll candidates before 2.5s are rejected by hook protection."""
    catalog = _make_sample_catalog(tmp_path)
    matcher = BrollMatcher(catalog=catalog, hook_protection=2.5)

    asset = catalog.get_asset("broll_tech")
    candidates = [
        CandidateOverlay(
            broll_id="broll_tech",
            timeline_start=1.0,  # Violates hook protection (< 2.5)
            duration=2.0,
            score=1.0,
            matched_keyword="coding",
            asset=asset,
        ),
        CandidateOverlay(
            broll_id="broll_tech",
            timeline_start=3.0,  # Valid (>= 2.5)
            duration=2.0,
            score=1.0,
            matched_keyword="python",
            asset=asset,
        ),
    ]

    overlays = matcher._enforce_pacing_constraints(candidates, total_duration=30.0)
    assert len(overlays) == 1
    assert overlays[0].timeline_start >= 2.5


def test_tail_protection_rule(tmp_path: Path):
    """Verify B-roll candidates exceeding total_duration - tail_protection are rejected."""
    catalog = _make_sample_catalog(tmp_path)
    matcher = BrollMatcher(catalog=catalog, tail_protection=1.0)
    asset = catalog.get_asset("broll_tech")

    candidates = [
        CandidateOverlay(
            broll_id="broll_tech",
            timeline_start=28.0,  # 28.0 + 2.0 = 30.0 > 29.0
            duration=2.0,
            score=1.0,
            matched_keyword="coding",
            asset=asset,
        ),
    ]
    overlays = matcher._enforce_pacing_constraints(candidates, total_duration=30.0)
    assert len(overlays) == 0


def test_spacing_cooldown_rule(tmp_path: Path):
    """Verify successive B-roll overlays enforce at least 5.0s of spacing."""
    catalog = _make_sample_catalog(tmp_path)
    matcher = BrollMatcher(catalog=catalog, min_spacing=5.0)
    tech = catalog.get_asset("broll_tech")
    money = catalog.get_asset("broll_money")

    candidates = [
        CandidateOverlay(broll_id="broll_tech", timeline_start=3.0, duration=2.0, score=1.0, matched_keyword="code", asset=tech),
        CandidateOverlay(broll_id="broll_money", timeline_start=6.0, duration=2.0, score=1.0, matched_keyword="money", asset=money), # end 5.0 + 5.0 = 10.0 > 6.0 (rejected!)
        CandidateOverlay(broll_id="broll_money", timeline_start=11.0, duration=2.0, score=1.0, matched_keyword="wealth", asset=money), # valid (11.0 >= 5.0 + 5.0)
    ]

    overlays = matcher._enforce_pacing_constraints(candidates, total_duration=60.0)
    assert len(overlays) == 2
    assert overlays[0].timeline_start == 3.0
    assert overlays[1].timeline_start == 11.0
    assert (overlays[1].timeline_start - overlays[0].timeline_end) >= 5.0


def test_budget_cap_rule(tmp_path: Path):
    """Verify total B-roll duration does not exceed 20% of video duration."""
    catalog = _make_sample_catalog(tmp_path)
    # Total duration = 20.0s -> max 20% is 4.0s
    matcher = BrollMatcher(catalog=catalog, max_coverage_ratio=0.20)
    tech = catalog.get_asset("broll_tech")
    money = catalog.get_asset("broll_money")
    geo = catalog.get_asset("broll_geo")

    candidates = [
        CandidateOverlay(broll_id="broll_tech", timeline_start=3.0, duration=2.5, score=1.0, matched_keyword="code", asset=tech),
        CandidateOverlay(broll_id="broll_money", timeline_start=11.0, duration=2.5, score=1.0, matched_keyword="money", asset=money), # 2.5 + 2.5 = 5.0 > 4.0 (rejected!)
    ]

    overlays = matcher._enforce_pacing_constraints(candidates, total_duration=20.0)
    total_overlay_dur = sum(o.duration for o in overlays)
    assert total_overlay_dur <= (0.20 * 20.0)


def test_ollama_fallback_to_heuristic(tmp_path: Path):
    """Verify failure in Ollama provider seamlessly triggers heuristic fallback."""
    catalog = _make_sample_catalog(tmp_path)

    failing_provider = MagicMock(spec=BaseProvider)
    failing_provider.is_available.return_value = True
    failing_provider.generate_json.side_effect = RuntimeError("Ollama connection timeout")

    matcher = BrollMatcher(provider=failing_provider, catalog=catalog)

    plan = EditingPlan(
        plan_id="plan_001",
        job_id="job_001",
        profile="dynamic_shorts",
        aspect_ratio="9:16",
        target_duration_seconds=30.0,
        hook_summary="Opening hook",
        edit_rationale="Fast pacing",
        clips=[Clip(clip_id="c001", source_start=0.0, source_end=30.0, order=1, reason="Full beat")],
    )
    transcript = Transcript(
        language="en",
        duration=30.0,
        segments=[
            TranscriptSegment(
                id=0,
                start=3.0,
                end=6.0,
                text="coding software and developer tools",
                words=[
                    TranscriptWord(word="coding", start=3.5, end=4.5, probability=0.99),
                    TranscriptWord(word="software", start=4.6, end=5.5, probability=0.99),
                ],
            )
        ],
    )

    overlays = matcher.match_broll(plan, transcript)
    assert len(overlays) >= 1
    assert overlays[0].broll_id == "broll_tech"
    assert overlays[0].timeline_start >= 2.5


def test_empty_catalog_and_missing_files_handled_gracefully(tmp_path: Path):
    """Verify empty catalog and missing files return empty overlay list without errors."""
    empty_catalog = BrollCatalog(index_path=tmp_path / "non_existent.json")
    matcher = BrollMatcher(catalog=empty_catalog)

    plan = EditingPlan(
        plan_id="plan_002",
        job_id="job_002",
        profile="dynamic_shorts",
        aspect_ratio="9:16",
        target_duration_seconds=30.0,
        hook_summary="Opening hook",
        edit_rationale="Fast pacing",
        clips=[Clip(clip_id="c001", source_start=0.0, source_end=30.0, order=1, reason="Beat 1")],
    )
    transcript = Transcript(language="en", duration=30.0, segments=[])

    overlays = matcher.match_broll(plan, transcript)
    assert overlays == []
