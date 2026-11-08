"""
tests/test_empirical_challenger_m3_2.py — Empirical Challenger Test Suite for Feature 10.

Adversarially tests and stress-tests:
1. Hook Protection Challenge (0 overlays scheduled before 2.5s).
2. Tail Protection Challenge (0 overlays extending within 1.0s of video end).
3. Spacing Cooldown Challenge (start_{i+1} >= end_i + 5.0s).
4. Duration Bounds Challenge (1.5s <= duration <= 3.5s).
5. Budget Cap Challenge (total duration <= 20% of video duration, count <= 5).
6. Hostile Transcripts (empty, 1-word, non-ASCII emojis/special chars, no matching tags).
7. Coordinate Translation Under Pauses (trimmed silences, non-monotonic source intervals, no temporal inversion).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from app.analysis.broll_matcher import BrollMatcher, CandidateOverlay, TimelineWord
from app.analysis.broll_pool import BrollAsset, BrollCatalog
from app.schemas.editing_plan import BrollOverlay, Clip, EditingPlan
from app.schemas.transcript import Transcript, TranscriptSegment, TranscriptWord


def _create_mock_broll_catalog(tmp_path: Path) -> BrollCatalog:
    """Create a temporary catalog with mock test media files spanning multiple categories."""
    broll_dir = tmp_path / "mock_broll"
    broll_dir.mkdir(parents=True, exist_ok=True)

    tech_file = broll_dir / "tech.mp4"
    tech_file.write_bytes(b"dummy_broll_tech")
    money_file = broll_dir / "money.mp4"
    money_file.write_bytes(b"dummy_broll_money")
    geo_file = broll_dir / "geo.mp4"
    geo_file.write_bytes(b"dummy_broll_geo")
    warning_file = broll_dir / "warning.mp4"
    warning_file.write_bytes(b"dummy_broll_warning")
    law_file = broll_dir / "law.mp4"
    law_file.write_bytes(b"dummy_broll_law")
    abstract_file = broll_dir / "abstract.mp4"
    abstract_file.write_bytes(b"dummy_broll_abstract")

    index_data = {
        "version": "1.0",
        "description": "Mock catalog for adversarial testing",
        "assets": [
            {
                "id": "broll_tech_01",
                "filename": "tech.mp4",
                "category": "technology_cyber",
                "tags": ["coding", "python", "developer", "software", "apple silicon", "algorithm"],
                "duration": 5.0,
                "description": "Tech code matrix",
            },
            {
                "id": "broll_money_01",
                "filename": "money.mp4",
                "category": "finance_money",
                "tags": ["finance", "money", "growth", "stock", "crypto", "wealth", "market"],
                "duration": 5.0,
                "description": "Financial stock chart",
            },
            {
                "id": "broll_geopolitics_01",
                "filename": "geo.mp4",
                "category": "geopolitics_conflict",
                "tags": ["military", "missiles", "radar", "war", "conflict", "iran", "attack"],
                "duration": 5.0,
                "description": "Tactical radar grid",
            },
            {
                "id": "broll_warning_01",
                "filename": "warning.mp4",
                "category": "warning_trap",
                "tags": ["warning", "danger", "trap", "alert", "threat", "risk"],
                "duration": 5.0,
                "description": "Hazard strobe pulse",
            },
            {
                "id": "broll_law_01",
                "filename": "law.mp4",
                "category": "law_justice",
                "tags": ["law", "court", "legal", "justice", "verdict", "contract"],
                "duration": 5.0,
                "description": "Scales of justice",
            },
            {
                "id": "broll_abstract_01",
                "filename": "abstract.mp4",
                "category": "abstract_motion",
                "tags": ["abstract", "speed", "motion", "energy", "hypnotic", "tunnel"],
                "duration": 5.0,
                "description": "Kinetic tunnel vortex",
            },
        ],
    }
    index_file = broll_dir / "broll_index.json"
    index_file.write_text(json.dumps(index_data), encoding="utf-8")
    return BrollCatalog(index_path=index_file)


# ==============================================================================
# Challenge 1: Hook Protection Challenge
# ==============================================================================


def test_hook_protection_challenge_early_keywords_rejected(tmp_path: Path):
    """
    HOOK PROTECTION CHALLENGE:
    Generate transcripts where high-scoring keywords appear strictly in [0.0s, 2.4s].
    Assert 0 overlays are scheduled before 2.5s.
    """
    catalog = _create_mock_broll_catalog(tmp_path)
    matcher = BrollMatcher(catalog=catalog, hook_protection=2.5)

    # Transcript with high-scoring keywords only in the first 2.4 seconds
    early_words = [
        TranscriptWord(word="warning", start=0.1, end=0.5, probability=0.99),
        TranscriptWord(word="danger", start=0.6, end=1.0, probability=0.99),
        TranscriptWord(word="coding", start=1.2, end=1.6, probability=0.99),
        TranscriptWord(word="software", start=1.7, end=2.1, probability=0.99),
        TranscriptWord(word="money", start=2.2, end=2.4, probability=0.99),
    ]
    transcript = Transcript(
        language="en",
        duration=30.0,
        segments=[TranscriptSegment(id=0, start=0.0, end=2.5, text="warning danger coding software money", words=early_words)],
    )

    plan = EditingPlan(
        plan_id="plan_hook_1",
        job_id="job_hook_1",
        profile="dynamic_shorts",
        aspect_ratio="9:16",
        target_duration_seconds=30.0,
        hook_summary="Hook intro",
        edit_rationale="Retention test",
        clips=[Clip(clip_id="c1", source_start=0.0, source_end=30.0, order=1, reason="Full")],
    )

    overlays = matcher.match_broll(plan, transcript)
    # Strictly zero overlays scheduled because all keywords are before hook threshold
    assert len(overlays) == 0, f"Expected 0 overlays scheduled before 2.5s, got {len(overlays)}"

    # Second case: Transcript contains both early keywords (<2.5s) AND later keywords (>2.5s)
    mixed_words = [
        TranscriptWord(word="coding", start=0.5, end=1.0, probability=0.99),
        TranscriptWord(word="software", start=1.5, end=2.0, probability=0.99),
        TranscriptWord(word="crypto", start=6.0, end=6.8, probability=0.99),
        TranscriptWord(word="growth", start=14.0, end=14.8, probability=0.99),
    ]
    mixed_transcript = Transcript(
        language="en",
        duration=30.0,
        segments=[TranscriptSegment(id=0, start=0.0, end=20.0, text="coding software crypto growth", words=mixed_words)],
    )

    mixed_overlays = matcher.match_broll(plan, mixed_transcript)
    assert len(mixed_overlays) > 0, "Should have scheduled overlays for later keywords"
    for o in mixed_overlays:
        assert o.timeline_start >= 2.5, f"Overlay {o.broll_id} scheduled at {o.timeline_start}s (< 2.5s hook protection)"


def test_hook_protection_boundary_exact_threshold(tmp_path: Path):
    """
    Verify boundary condition around hook protection threshold (2.5s).
    """
    catalog = _create_mock_broll_catalog(tmp_path)
    matcher = BrollMatcher(catalog=catalog, hook_protection=2.5)

    asset = catalog.get_asset("broll_tech_01")
    candidates = [
        CandidateOverlay(broll_id="broll_tech_01", timeline_start=2.49, duration=2.0, score=1.0, matched_keyword="code", asset=asset),
        CandidateOverlay(broll_id="broll_tech_01", timeline_start=2.50, duration=2.0, score=1.0, matched_keyword="code", asset=asset),
    ]

    overlays = matcher._enforce_pacing_constraints(candidates, total_duration=30.0)
    assert len(overlays) == 1
    assert overlays[0].timeline_start == 2.50


# ==============================================================================
# Challenge 2: Tail Protection Challenge
# ==============================================================================


def test_tail_protection_challenge_late_keywords_rejected(tmp_path: Path):
    """
    TAIL PROTECTION CHALLENGE:
    Generate transcript with keywords situated right before video end.
    Assert 0 overlays extend within 1.0s of total video end.
    """
    catalog = _create_mock_broll_catalog(tmp_path)
    total_duration = 30.0
    matcher = BrollMatcher(catalog=catalog, tail_protection=1.0)

    # Keywords right at the end of the video: 28.0s, 28.8s, 29.5s
    late_words = [
        TranscriptWord(word="court", start=28.0, end=28.5, probability=0.99),
        TranscriptWord(word="verdict", start=28.8, end=29.3, probability=0.99),
        TranscriptWord(word="justice", start=29.4, end=29.9, probability=0.99),
    ]
    transcript = Transcript(
        language="en",
        duration=total_duration,
        segments=[TranscriptSegment(id=0, start=28.0, end=30.0, text="court verdict justice", words=late_words)],
    )

    plan = EditingPlan(
        plan_id="plan_tail_1",
        job_id="job_tail_1",
        profile="dynamic_shorts",
        aspect_ratio="9:16",
        target_duration_seconds=total_duration,
        hook_summary="Tail test",
        edit_rationale="Retention test",
        clips=[Clip(clip_id="c1", source_start=0.0, source_end=total_duration, order=1, reason="Full")],
    )

    overlays = matcher.match_broll(plan, transcript)
    # Since minimum duration is 1.5s, candidate at 28.0s ends at >= 29.35s > 29.0s (violates tail protection)
    assert len(overlays) == 0, f"Expected 0 overlays right before video end, got {len(overlays)}"

    # Mixed test: Keywords at 5.0s, 15.0s, and 28.5s
    mixed_words = [
        TranscriptWord(word="coding", start=5.0, end=5.8, probability=0.99),
        TranscriptWord(word="wealth", start=15.0, end=15.8, probability=0.99),
        TranscriptWord(word="missiles", start=28.5, end=29.2, probability=0.99),
    ]
    mixed_transcript = Transcript(
        language="en",
        duration=total_duration,
        segments=[TranscriptSegment(id=0, start=0.0, end=total_duration, text="coding wealth missiles", words=mixed_words)],
    )

    mixed_overlays = matcher.match_broll(plan, mixed_transcript)
    assert len(mixed_overlays) >= 1
    for o in mixed_overlays:
        assert o.timeline_end <= (total_duration - 1.0), (
            f"Overlay {o.broll_id} extends to {o.timeline_end}s, which violates tail protection threshold {total_duration - 1.0}s"
        )


def test_tail_protection_boundary_exact_threshold(tmp_path: Path):
    """
    Verify boundary condition at total_duration - tail_protection.
    """
    catalog = _create_mock_broll_catalog(tmp_path)
    matcher = BrollMatcher(catalog=catalog, min_duration=1.5, tail_protection=1.0)
    asset = catalog.get_asset("broll_tech_01")
    total_duration = 30.0
    cutoff = total_duration - 1.0  # 29.0s

    # Candidate 1: 27.50 + 1.50 = 29.00 -> exactly on boundary -> ALLOWED
    # Candidate 2: 27.51 + 1.50 = 29.01 -> exceeds 29.00 -> REJECTED
    candidates = [
        CandidateOverlay(broll_id="broll_tech_01", timeline_start=27.50, duration=1.5, score=1.0, matched_keyword="code", asset=asset),
        CandidateOverlay(broll_id="broll_tech_01", timeline_start=27.51, duration=1.5, score=1.0, matched_keyword="code", asset=asset),
    ]

    overlays = matcher._enforce_pacing_constraints(candidates, total_duration=total_duration)
    assert len(overlays) == 1
    assert overlays[0].timeline_start == 27.50
    assert overlays[0].timeline_end <= cutoff


# ==============================================================================
# Challenge 3: Spacing Cooldown Challenge
# ==============================================================================


def test_spacing_cooldown_challenge_continuous_high_density(tmp_path: Path):
    """
    SPACING COOLDOWN CHALLENGE:
    Generate continuous high-density keyword transcripts (e.g. keywords every 0.5s).
    Assert that every overlay pair (i, i+1) satisfies start_{i+1} >= end_i + 5.0s.
    """
    catalog = _create_mock_broll_catalog(tmp_path)
    matcher = BrollMatcher(catalog=catalog, min_spacing=5.0)
    total_duration = 60.0

    # Dense keywords firing every 0.5 seconds from 2.0s to 55.0s
    tags_pool = ["coding", "software", "money", "finance", "radar", "military", "court", "justice", "warning", "abstract"]
    dense_words: List[TranscriptWord] = []
    t = 2.0
    idx = 0
    while t < 55.0:
        word = tags_pool[idx % len(tags_pool)]
        dense_words.append(TranscriptWord(word=word, start=t, end=t + 0.4, probability=0.99))
        t += 0.5
        idx += 1

    transcript = Transcript(
        language="en",
        duration=total_duration,
        segments=[TranscriptSegment(id=0, start=2.0, end=55.0, text="dense stream", words=dense_words)],
    )

    plan = EditingPlan(
        plan_id="plan_spacing_1",
        job_id="job_spacing_1",
        profile="dynamic_shorts",
        aspect_ratio="9:16",
        target_duration_seconds=total_duration,
        hook_summary="Spacing test",
        edit_rationale="Retention test",
        clips=[Clip(clip_id="c1", source_start=0.0, source_end=total_duration, order=1, reason="Full")],
    )

    overlays = matcher.match_broll(plan, transcript)
    assert len(overlays) >= 3, f"Expected multiple overlays scheduled, got {len(overlays)}"

    # Check spacing cooldown across all adjacent pairs
    for i in range(len(overlays) - 1):
        prev_overlay = overlays[i]
        curr_overlay = overlays[i + 1]
        spacing = curr_overlay.timeline_start - prev_overlay.timeline_end
        assert spacing >= 5.0 - 1e-4, (
            f"Cooldown violation between overlay {i} ({prev_overlay.broll_id} ends at {prev_overlay.timeline_end}s) "
            f"and overlay {i+1} ({curr_overlay.broll_id} starts at {curr_overlay.timeline_start}s): spacing is {spacing:.3f}s (< 5.0s)"
        )


def test_spacing_cooldown_custom_spacing(tmp_path: Path):
    """
    Verify custom cooldown spacing (e.g. 7.5s) is strictly respected.
    """
    catalog = _create_mock_broll_catalog(tmp_path)
    custom_spacing = 7.5
    matcher = BrollMatcher(catalog=catalog, min_spacing=custom_spacing)
    tech = catalog.get_asset("broll_tech_01")
    money = catalog.get_asset("broll_money_01")

    candidates = [
        CandidateOverlay(broll_id="broll_tech_01", timeline_start=3.0, duration=2.0, score=1.0, matched_keyword="code", asset=tech),
        CandidateOverlay(broll_id="broll_money_01", timeline_start=8.0, duration=2.0, score=1.0, matched_keyword="money", asset=money),  # 8.0 < 3.0 + 2.0 + 7.5 = 12.5 (REJECT)
        CandidateOverlay(broll_id="broll_money_01", timeline_start=12.5, duration=2.0, score=1.0, matched_keyword="money", asset=money), # 12.5 >= 12.5 (ACCEPT)
    ]

    overlays = matcher._enforce_pacing_constraints(candidates, total_duration=60.0)
    assert len(overlays) == 2
    assert overlays[0].timeline_start == 3.0
    assert overlays[1].timeline_start == 12.5
    assert (overlays[1].timeline_start - overlays[0].timeline_end) >= custom_spacing


# ==============================================================================
# Challenge 4: Duration Bounds Challenge
# ==============================================================================


def test_duration_bounds_challenge_strict_clamping(tmp_path: Path):
    """
    DURATION BOUNDS CHALLENGE:
    Assert all matched overlays strictly satisfy 1.5s <= duration <= 3.5s.
    """
    catalog = _create_mock_broll_catalog(tmp_path)
    matcher = BrollMatcher(catalog=catalog, min_duration=1.5, max_duration=3.5, default_duration=2.2)

    tech = catalog.get_asset("broll_tech_01")
    money = catalog.get_asset("broll_money_01")
    geo = catalog.get_asset("broll_geopolitics_01")
    warning = catalog.get_asset("broll_warning_01")
    law = catalog.get_asset("broll_law_01")

    # Supply candidates with extreme durations using distinct assets: 0.2s, 0.9s, 2.2s, 2.5s, 12.0s
    candidates = [
        CandidateOverlay(broll_id="broll_tech_01", timeline_start=3.0, duration=0.2, score=1.0, matched_keyword="code", asset=tech),
        CandidateOverlay(broll_id="broll_money_01", timeline_start=10.0, duration=0.9, score=1.0, matched_keyword="money", asset=money),
        CandidateOverlay(broll_id="broll_geopolitics_01", timeline_start=17.0, duration=2.2, score=1.0, matched_keyword="radar", asset=geo),
        CandidateOverlay(broll_id="broll_warning_01", timeline_start=25.0, duration=2.5, score=1.0, matched_keyword="danger", asset=warning),
        CandidateOverlay(broll_id="broll_law_01", timeline_start=35.0, duration=12.0, score=1.0, matched_keyword="justice", asset=law),
    ]

    # Total duration = 70.0s -> 20% budget is 14.0s (1.5 + 1.5 + 2.2 + 2.5 + 3.5 = 11.2s <= 14.0s)
    overlays = matcher._enforce_pacing_constraints(candidates, total_duration=70.0)
    assert len(overlays) == 5

    for idx, o in enumerate(overlays):
        dur = round(o.duration, 4)
        assert 1.5 <= dur <= 3.5, (
            f"Overlay {idx} duration {dur:.2f}s violates bounds [1.5s, 3.5s]"
        )

    # Candidate with 0.2s must be clamped up to exactly 1.5s
    assert round(overlays[0].duration, 2) == 1.50
    # Candidate with 0.9s must be clamped up to exactly 1.5s
    assert round(overlays[1].duration, 2) == 1.50
    # Candidate with 2.2s remains 2.2s
    assert round(overlays[2].duration, 2) == 2.20
    # Candidate with 2.5s remains 2.5s
    assert round(overlays[3].duration, 2) == 2.50
    # Candidate with 12.0s must be clamped down to exactly 3.5s
    assert round(overlays[4].duration, 2) == 3.50


# ==============================================================================
# Challenge 5: Budget Cap Challenge
# ==============================================================================


def test_budget_cap_challenge_60s_video(tmp_path: Path):
    """
    BUDGET CAP CHALLENGE:
    In a 60-second video, assert total overlay duration never exceeds 20% (12.0s)
    and overlay count never exceeds 5.
    """
    catalog = _create_mock_broll_catalog(tmp_path)
    # Default: max_coverage_ratio=0.20, max_overlays=5
    matcher = BrollMatcher(catalog=catalog, max_coverage_ratio=0.20, max_overlays=5)
    total_duration = 60.0
    budget_cap = 0.20 * total_duration  # 12.0s

    # Generate 10 spaced keyword events across the video
    keywords = ["coding", "money", "radar", "warning", "court", "abstract", "developer", "crypto", "missiles", "threat"]
    words: List[TranscriptWord] = []
    for i, kw in enumerate(keywords):
        t_start = 3.0 + (i * 5.5)  # 3.0, 8.5, 14.0, 19.5, 25.0, 30.5, 36.0, 41.5, 47.0, 52.5
        words.append(TranscriptWord(word=kw, start=t_start, end=t_start + 0.8, probability=0.99))

    transcript = Transcript(
        language="en",
        duration=total_duration,
        segments=[TranscriptSegment(id=0, start=0.0, end=total_duration, text="budget test stream", words=words)],
    )

    plan = EditingPlan(
        plan_id="plan_budget_60",
        job_id="job_budget_60",
        profile="dynamic_shorts",
        aspect_ratio="9:16",
        target_duration_seconds=total_duration,
        hook_summary="Budget test",
        edit_rationale="Retention test",
        clips=[Clip(clip_id="c1", source_start=0.0, source_end=total_duration, order=1, reason="Full")],
    )

    overlays = matcher.match_broll(plan, transcript)
    total_overlay_dur = sum(o.duration for o in overlays)

    # Strict assertion 1: overlay count never exceeds 5
    assert len(overlays) <= 5, f"Overlay count {len(overlays)} exceeds max_overlays cap of 5"

    # Strict assertion 2: total overlay duration never exceeds 20% (12.0s)
    assert total_overlay_dur <= budget_cap, (
        f"Total overlay duration {total_overlay_dur:.2f}s exceeds 20% budget cap {budget_cap:.2f}s"
    )


def test_budget_cap_short_video_tight_limit(tmp_path: Path):
    """
    Verify budget cap on a short 20.0s video (20% budget = 4.0s).
    Two 2.2s cutaways would be 4.4s (> 4.0s), so only 1 cutaway must be scheduled.
    """
    catalog = _create_mock_broll_catalog(tmp_path)
    matcher = BrollMatcher(catalog=catalog, default_duration=2.2, max_coverage_ratio=0.20)
    total_duration = 20.0
    budget_cap = 0.20 * total_duration  # 4.0s

    tech = catalog.get_asset("broll_tech_01")
    money = catalog.get_asset("broll_money_01")

    candidates = [
        CandidateOverlay(broll_id="broll_tech_01", timeline_start=3.0, duration=2.2, score=1.0, matched_keyword="code", asset=tech),
        CandidateOverlay(broll_id="broll_money_01", timeline_start=11.0, duration=2.2, score=1.0, matched_keyword="money", asset=money),
    ]

    overlays = matcher._enforce_pacing_constraints(candidates, total_duration=total_duration)
    total_dur = sum(o.duration for o in overlays)

    assert len(overlays) == 1, "Only 1 overlay should fit within the 4.0s budget cap"
    assert total_dur <= budget_cap


# ==============================================================================
# Challenge 6: Hostile Transcripts Challenge
# ==============================================================================


def test_hostile_transcripts_challenge_resilience(tmp_path: Path):
    """
    HOSTILE TRANSCRIPTS CHALLENGE:
    Test empty transcript, 1-word transcript, transcript with non-ASCII emojis/special characters,
    and transcript with no matching tags.
    Assert zero unhandled exceptions.
    """
    catalog = _create_mock_broll_catalog(tmp_path)
    matcher = BrollMatcher(catalog=catalog)

    plan = EditingPlan(
        plan_id="plan_hostile",
        job_id="job_hostile",
        profile="dynamic_shorts",
        aspect_ratio="9:16",
        target_duration_seconds=30.0,
        hook_summary="Hostile test",
        edit_rationale="Retention test",
        clips=[Clip(clip_id="c1", source_start=0.0, source_end=30.0, order=1, reason="Full")],
    )

    # 1. None transcript
    res_none = matcher.match_broll(plan, None)
    assert res_none == []

    # 2. Empty transcript (0 segments)
    empty_transcript = Transcript(language="en", duration=30.0, segments=[])
    res_empty = matcher.match_broll(plan, empty_transcript)
    assert res_empty == []

    # 3. 1-word transcript: Stopword ("the")
    stopword_transcript = Transcript(
        language="en",
        duration=30.0,
        segments=[TranscriptSegment(id=0, start=3.0, end=4.0, text="the", words=[TranscriptWord(word="the", start=3.0, end=4.0, probability=0.99)])],
    )
    res_stopword = matcher.match_broll(plan, stopword_transcript)
    assert res_stopword == []

    # 4. 1-word transcript: Keyword ("coding")
    oneword_transcript = Transcript(
        language="en",
        duration=30.0,
        segments=[TranscriptSegment(id=0, start=3.5, end=4.5, text="coding", words=[TranscriptWord(word="coding", start=3.5, end=4.5, probability=0.99)])],
    )
    res_oneword = matcher.match_broll(plan, oneword_transcript)
    assert len(res_oneword) == 1
    assert res_oneword[0].broll_id == "broll_tech_01"

    # 5. Non-ASCII emojis, unicode control characters, RTL glyphs, Asian characters
    hostile_text = (
        "🔥🚀🤖 ⚠️\u200b\u200d\u26a0\ufe0f \x00\x01\x1f "
        "\u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u4e2d\u6587 "
        "!@#$%^&*()_+-=[]{}|;':,./<>? coding 💰"
    )
    hostile_words = [
        TranscriptWord(word="🔥🚀🤖", start=3.0, end=3.4, probability=0.8),
        TranscriptWord(word="⚠️\u200b\u200d\u26a0\ufe0f", start=3.5, end=3.9, probability=0.8),
        TranscriptWord(word="\x00\x01\x1f", start=4.0, end=4.4, probability=0.8),
        TranscriptWord(word="\u4e2d\u6587", start=4.5, end=4.9, probability=0.8),
        TranscriptWord(word="coding", start=5.0, end=5.5, probability=0.99),
        TranscriptWord(word="💰", start=5.6, end=6.0, probability=0.8),
    ]
    hostile_transcript = Transcript(
        language="en",
        duration=30.0,
        segments=[TranscriptSegment(id=0, start=3.0, end=6.0, text=hostile_text, words=hostile_words)],
    )
    res_hostile = matcher.match_broll(plan, hostile_transcript)
    # Zero unhandled exceptions; successfully matches "coding" despite surrounding hostile unicode
    assert isinstance(res_hostile, list)
    assert len(res_hostile) >= 1
    assert res_hostile[0].broll_id == "broll_tech_01"

    # 6. Transcript with zero matching tags
    no_match_text = "the quick brown fox jumps over the lazy dog zebra banana cucumber elephant"
    no_match_words = [
        TranscriptWord(word=w, start=3.0 + i * 0.5, end=3.4 + i * 0.5, probability=0.99)
        for i, w in enumerate(no_match_text.split())
    ]
    no_match_transcript = Transcript(
        language="en",
        duration=30.0,
        segments=[TranscriptSegment(id=0, start=3.0, end=10.0, text=no_match_text, words=no_match_words)],
    )
    res_no_match = matcher.match_broll(plan, no_match_transcript)
    assert res_no_match == []

    # 7. Transcript with segment-level text only (no word timestamps)
    seg_only_transcript = Transcript(
        language="en",
        duration=30.0,
        segments=[
            TranscriptSegment(id=0, start=4.0, end=8.0, text="artificial intelligence and python algorithm development", words=[]),
        ],
    )
    res_seg_only = matcher.match_broll(plan, seg_only_transcript)
    assert isinstance(res_seg_only, list)
    assert len(res_seg_only) >= 1


# ==============================================================================
# Challenge 7: Coordinate Translation Under Pauses & Non-Monotonic Clips
# ==============================================================================


def test_coordinate_translation_under_pauses_and_non_monotonic_clips():
    """
    COORDINATE TRANSLATION UNDER PAUSES CHALLENGE:
    Create multiple clips with non-monotonic source intervals (simulating trimmed silences / montage cuts)
    and verify that `_map_transcript_to_timeline` correctly maps word timestamps without temporal inversion.
    """
    matcher = BrollMatcher()

    # Scenario: 3 clips with non-monotonic source jumps and trimmed silences:
    # Clip 1: source [30.0, 40.0] (dur 10.0s) -> timeline [0.0, 10.0]
    # (Gap in source: 40.0 to 50.0 is trimmed silence)
    # Clip 2: source [5.0, 15.0] (dur 10.0s) -> timeline [10.0, 20.0] (non-monotonic jump back in source time!)
    # (Gap in source: 15.0 to 60.0 is trimmed silence)
    # Clip 3: source [60.0, 70.0] (dur 10.0s) -> timeline [20.0, 30.0]
    clips = [
        Clip(clip_id="c001", source_start=30.0, source_end=40.0, order=1, reason="Act 1"),
        Clip(clip_id="c002", source_start=5.0, source_end=15.0, order=2, reason="Flashback Act 2"),
        Clip(clip_id="c003", source_start=60.0, source_end=70.0, order=3, reason="Climax Act 3"),
    ]

    words = [
        # In Clip 1:
        TranscriptWord(word="alpha", start=31.0, end=32.0, probability=0.99),
        TranscriptWord(word="beta", start=35.5, end=36.5, probability=0.99),
        # In trimmed silence between Clip 1 and Clip 2 (source 40-50):
        TranscriptWord(word="dropped_silence_1", start=42.0, end=43.0, probability=0.99),
        TranscriptWord(word="dropped_silence_2", start=48.0, end=49.0, probability=0.99),
        # In Clip 2 (source 5-15):
        TranscriptWord(word="gamma", start=7.0, end=8.0, probability=0.99),
        TranscriptWord(word="delta", start=12.0, end=13.0, probability=0.99),
        # In trimmed silence between Clip 2 and Clip 3 (source 15-60):
        TranscriptWord(word="dropped_silence_3", start=25.0, end=26.0, probability=0.99),
        # In Clip 3 (source 60-70):
        TranscriptWord(word="epsilon", start=62.0, end=63.0, probability=0.99),
        TranscriptWord(word="omega", start=68.0, end=69.0, probability=0.99),
    ]

    transcript = Transcript(
        language="en",
        duration=80.0,
        segments=[TranscriptSegment(id=0, start=0.0, end=80.0, text="words", words=words)],
    )

    timeline_words = matcher._map_transcript_to_timeline(clips, transcript)
    clean_words = [w.clean_word for w in timeline_words]

    # Verification 1: Dropped silence words are strictly excluded
    assert "droppedsilence1" not in clean_words
    assert "droppedsilence2" not in clean_words
    assert "droppedsilence3" not in clean_words

    # Verification 2: Expected words are all present
    assert clean_words == ["alpha", "beta", "gamma", "delta", "epsilon", "omega"]

    # Verification 3: Exact mathematical translation:
    # alpha: clip 1 offset 0.0 + (31.0 - 30.0) = 1.0s
    assert pytest.approx(timeline_words[0].timeline_start, abs=1e-3) == 1.0
    assert pytest.approx(timeline_words[0].timeline_end, abs=1e-3) == 2.0

    # beta: clip 1 offset 0.0 + (35.5 - 30.0) = 5.5s
    assert pytest.approx(timeline_words[1].timeline_start, abs=1e-3) == 5.5
    assert pytest.approx(timeline_words[1].timeline_end, abs=1e-3) == 6.5

    # gamma: clip 2 offset 10.0 + (7.0 - 5.0) = 12.0s
    assert pytest.approx(timeline_words[2].timeline_start, abs=1e-3) == 12.0
    assert pytest.approx(timeline_words[2].timeline_end, abs=1e-3) == 13.0

    # delta: clip 2 offset 10.0 + (12.0 - 5.0) = 17.0s
    assert pytest.approx(timeline_words[3].timeline_start, abs=1e-3) == 17.0
    assert pytest.approx(timeline_words[3].timeline_end, abs=1e-3) == 18.0

    # epsilon: clip 3 offset 20.0 + (62.0 - 60.0) = 22.0s
    assert pytest.approx(timeline_words[4].timeline_start, abs=1e-3) == 22.0
    assert pytest.approx(timeline_words[4].timeline_end, abs=1e-3) == 23.0

    # omega: clip 3 offset 20.0 + (68.0 - 60.0) = 28.0s
    assert pytest.approx(timeline_words[5].timeline_start, abs=1e-3) == 28.0
    assert pytest.approx(timeline_words[5].timeline_end, abs=1e-3) == 29.0

    # Verification 4: NO TEMPORAL INVERSION
    # Timeline timestamps must be strictly monotonic and each word must have start <= end
    for i in range(len(timeline_words)):
        tw = timeline_words[i]
        assert tw.timeline_start <= tw.timeline_end, (
            f"Word '{tw.word}' has inverted boundaries: start={tw.timeline_start} > end={tw.timeline_end}"
        )
        if i > 0:
            prev_tw = timeline_words[i - 1]
            assert tw.timeline_start >= prev_tw.timeline_start, (
                f"Temporal inversion detected between '{prev_tw.word}' ({prev_tw.timeline_start}) "
                f"and '{tw.word}' ({tw.timeline_start})"
            )


def test_coordinate_translation_clip_boundary_clamping():
    """
    Verify words that straddle clip boundaries are safely clipped to the active clip window.
    """
    matcher = BrollMatcher()
    clips = [
        Clip(clip_id="c001", source_start=10.0, source_end=20.0, order=1, reason="Active window"),
    ]
    straddling_words = [
        # Straddles start: starts before clip (9.0), ends inside clip (12.0)
        TranscriptWord(word="startstraddle", start=9.0, end=12.0, probability=0.99),
        # Straddles end: starts inside clip (18.0), ends after clip (21.5)
        TranscriptWord(word="endstraddle", start=18.0, end=21.5, probability=0.99),
    ]
    transcript = Transcript(
        language="en",
        duration=30.0,
        segments=[TranscriptSegment(id=0, start=8.0, end=22.0, text="straddles", words=straddling_words)],
    )

    timeline_words = matcher._map_transcript_to_timeline(clips, transcript)
    assert len(timeline_words) == 2

    # startstraddle: clamped to [10.0, 12.0] -> timeline [0.0, 2.0]
    assert pytest.approx(timeline_words[0].timeline_start, abs=1e-3) == 0.0
    assert pytest.approx(timeline_words[0].timeline_end, abs=1e-3) == 2.0

    # endstraddle: clamped to [18.0, 20.0] -> timeline [8.0, 10.0]
    assert pytest.approx(timeline_words[1].timeline_start, abs=1e-3) == 8.0
    assert pytest.approx(timeline_words[1].timeline_end, abs=1e-3) == 10.0


# ==============================================================================
# Challenge 8: Full Integrated Retentive Pipeline Verification
# ==============================================================================


def test_full_retentive_pipeline_compliance(tmp_path: Path):
    """
    Full end-to-end retentive pacing verification on conformed editing plan.
    Asserts all 6 viral retention invariant rules simultaneously.
    """
    catalog = _create_mock_broll_catalog(tmp_path)
    matcher = BrollMatcher(catalog=catalog)
    total_duration = 50.0

    clips = [
        Clip(clip_id="c1", source_start=0.0, source_end=20.0, order=1, reason="Beat 1"),
        Clip(clip_id="c2", source_start=30.0, source_end=60.0, order=2, reason="Beat 2"),  # 30s dur -> total 50.0s
    ]
    plan = EditingPlan(
        plan_id="plan_full_e2e",
        job_id="job_full_e2e",
        profile="dynamic_shorts",
        aspect_ratio="9:16",
        target_duration_seconds=total_duration,
        hook_summary="Full test",
        edit_rationale="Retention test",
        clips=clips,
    )

    # Keywords placed at various spots:
    # 1.0s (in hook -> must be rejected)
    # 5.0s (valid -> start ~ 4.85)
    # 8.0s (too close to 5.0s -> cooldown rejected)
    # 14.0s (valid -> start ~ 13.85)
    # In trimmed source silence 22.0s -> dropped by coordinate mapper
    # In clip 2: source 35.0s -> timeline 25.0s (valid -> start ~ 24.85)
    # In clip 2: source 45.0s -> timeline 35.0s (valid -> start ~ 34.85)
    # In clip 2: source 59.5s -> timeline 49.5s (in tail -> rejected)
    words = [
        TranscriptWord(word="warning", start=1.0, end=1.5, probability=0.99),
        TranscriptWord(word="coding", start=5.0, end=5.5, probability=0.99),
        TranscriptWord(word="software", start=8.0, end=8.5, probability=0.99),
        TranscriptWord(word="money", start=14.0, end=14.5, probability=0.99),
        TranscriptWord(word="radar", start=22.0, end=22.5, probability=0.99),  # Inside cut silence
        TranscriptWord(word="court", start=35.0, end=35.5, probability=0.99),
        TranscriptWord(word="abstract", start=45.0, end=45.5, probability=0.99),
        TranscriptWord(word="threat", start=59.5, end=59.8, probability=0.99),  # Inside protected tail
    ]
    transcript = Transcript(
        language="en",
        duration=65.0,
        segments=[TranscriptSegment(id=0, start=0.0, end=60.0, text="words", words=words)],
    )

    overlays = matcher.match_broll(plan, transcript)
    assert len(overlays) >= 3

    # Rule 1: Hook Protection
    for o in overlays:
        assert o.timeline_start >= 2.5, f"Hook violated: {o.timeline_start} < 2.5"

    # Rule 2: Tail Protection
    for o in overlays:
        assert o.timeline_end <= total_duration - 1.0, f"Tail violated: {o.timeline_end} > {total_duration - 1.0}"

    # Rule 3: Spacing Cooldown
    for i in range(len(overlays) - 1):
        spacing = overlays[i + 1].timeline_start - overlays[i].timeline_end
        assert spacing >= 5.0 - 1e-4, f"Spacing violated: {spacing} < 5.0"

    # Rule 4: Duration Bounds
    for o in overlays:
        assert 1.5 <= o.duration <= 3.5, f"Duration bounds violated: {o.duration}"

    # Rule 5: Budget Cap
    total_overlay_dur = sum(o.duration for o in overlays)
    assert total_overlay_dur <= 0.20 * total_duration, f"Budget cap violated: {total_overlay_dur} > {0.20 * total_duration}"
    assert len(overlays) <= 5, f"Count cap violated: {len(overlays)} > 5"

    # Rule 6: Pydantic Schema Validation
    plan.broll_overlays = overlays
    assert len(plan.broll_overlays) == len(overlays)
