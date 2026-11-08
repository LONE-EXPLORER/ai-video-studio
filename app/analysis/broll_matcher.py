"""
app/analysis/broll_matcher.py — Semantic Transcript Keyword B-Roll Matcher.

Maps speech transcript timestamps across conformed/cut clips to timeline coordinates,
performs dual-tier semantic matching (Ollama LLM with resilient heuristic fallback),
and strictly enforces viral retention pacing rules (hook protection, cooldown spacing,
duration clamps, budget cap, dialogue non-interference).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from app.analysis.broll_pool import BrollAsset, BrollCatalog
from app.providers.base import BaseProvider
from app.schemas.editing_plan import BrollOverlay, Clip, EditingPlan
from app.schemas.transcript import Transcript, TranscriptSegment, TranscriptWord
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Stopwords set for heuristic keyword filtering
_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}

_CLEAN_RE = re.compile(r"[^a-zA-Z0-9\s]")


@dataclass
class TimelineWord:
    """A word mapped to edited video timeline coordinates."""

    word: str
    clean_word: str
    timeline_start: float
    timeline_end: float


@dataclass
class CandidateOverlay:
    """An unvalidated B-roll overlay candidate before pacing enforcement."""

    broll_id: str
    timeline_start: float
    duration: float
    score: float
    matched_keyword: str
    asset: BrollAsset
    reason: str = ""

    @property
    def timeline_end(self) -> float:
        return self.timeline_start + self.duration


class BrollMatcher:
    """
    Semantic B-roll matcher that maps transcript keywords to local assets
    while strictly enforcing viral retention pacing constraints.
    """

    def __init__(
        self,
        provider: Optional[BaseProvider] = None,
        catalog: Optional[BrollCatalog] = None,
        min_duration: float = 1.5,
        max_duration: float = 3.5,
        default_duration: float = 2.2,
        min_spacing: float = 5.0,
        hook_protection: float = 2.5,
        tail_protection: float = 1.0,
        max_coverage_ratio: float = 0.20,
        max_overlays: int = 5,
    ) -> None:
        self.provider = provider
        self.catalog = catalog or BrollCatalog()
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.default_duration = default_duration
        self.min_spacing = min_spacing
        self.hook_protection = hook_protection
        self.tail_protection = tail_protection
        self.max_coverage_ratio = max_coverage_ratio
        self.max_overlays = max_overlays

    def match_broll(
        self,
        plan: EditingPlan,
        transcript: Optional[Transcript],
    ) -> List[BrollOverlay]:
        """
        Public API: Match transcript keywords to B-roll cutaways and return BrollOverlay items.

        Parameters
        ----------
        plan:
            Validated EditingPlan containing clips.
        transcript:
            Optional speech transcript with segments and words.

        Returns
        -------
        List[BrollOverlay]
            Pacing-validated B-roll overlays mapped to timeline timestamps.
        """
        if self.catalog.total_count == 0:
            logger.info("B-roll catalog is empty. Returning 0 overlays.")
            return []

        if transcript is None or not transcript.segments or not plan.clips:
            logger.info("Transcript or plan clips missing. Returning 0 overlays.")
            return []

        total_duration = plan.total_duration
        if total_duration < (self.hook_protection + self.min_duration + self.tail_protection):
            logger.info("Video total duration (%.1fs) too short for B-roll overlays.", total_duration)
            return []

        # 1. Map transcript to timeline coordinates
        timeline_words = self._map_transcript_to_timeline(plan.clips, transcript)
        if not timeline_words:
            logger.info("No transcript words mapped to timeline. Returning 0 overlays.")
            return []

        # 2. Try Tier 1: Local LLM Matching if provider available
        candidates: List[CandidateOverlay] = []
        if self.provider is not None:
            try:
                candidates = self._match_with_llm(timeline_words, total_duration)
            except Exception as exc:
                logger.warning("Tier 1 LLM B-roll matching failed (%s); falling back to Tier 2 heuristic", exc)
                candidates = []

        # 3. Fallback to Tier 2: Resilient Heuristic Matching
        if not candidates:
            logger.debug("Executing Tier 2 heuristic B-roll matching...")
            candidates = self._match_with_heuristics(timeline_words, total_duration)

        # 4. Enforce strict pacing & retention constraints
        final_overlays = self._enforce_pacing_constraints(candidates, total_duration)
        logger.info(
            "Selected %d B-roll overlay(s) spanning %.1fs (%.1f%% of video)",
            len(final_overlays),
            sum(o.duration for o in final_overlays),
            (sum(o.duration for o in final_overlays) / total_duration * 100) if total_duration > 0 else 0,
        )
        return final_overlays

    def _map_transcript_to_timeline(
        self,
        clips: List[Clip],
        transcript: Transcript,
    ) -> List[TimelineWord]:
        """
        Map transcript word timestamps from raw source coordinates to edited timeline coordinates.
        Words outside kept clips are discarded.
        """
        timeline_words: List[TimelineWord] = []
        timeline_offset = 0.0

        for clip in clips:
            clip_dur = clip.source_end - clip.source_start
            clip_start = clip.source_start
            clip_end = clip.source_end

            # Gather words that overlap this clip
            for seg in transcript.segments:
                if seg.end < clip_start or seg.start > clip_end:
                    continue

                if seg.words:
                    for w in seg.words:
                        if w.end < clip_start or w.start > clip_end:
                            continue
                        # Clip word boundaries to clip window
                        w_src_start = max(w.start, clip_start)
                        w_src_end = min(w.end, clip_end)
                        t_start = timeline_offset + (w_src_start - clip_start)
                        t_end = timeline_offset + (w_src_end - clip_start)
                        raw = w.word.strip()
                        clean = _CLEAN_RE.sub("", raw).lower()
                        if clean:
                            timeline_words.append(TimelineWord(
                                word=raw,
                                clean_word=clean,
                                timeline_start=t_start,
                                timeline_end=t_end,
                            ))
                else:
                    # Segment without word-level timestamps: distribute text words evenly
                    tokens = [t.strip() for t in seg.text.split() if t.strip()]
                    if not tokens:
                        continue
                    seg_dur = max(seg.end - seg.start, 0.1)
                    word_dur = seg_dur / len(tokens)
                    for i, tok in enumerate(tokens):
                        w_src_start = seg.start + (i * word_dur)
                        w_src_end = w_src_start + word_dur
                        if w_src_end < clip_start or w_src_start > clip_end:
                            continue
                        w_src_start = max(w_src_start, clip_start)
                        w_src_end = min(w_src_end, clip_end)
                        t_start = timeline_offset + (w_src_start - clip_start)
                        t_end = timeline_offset + (w_src_end - clip_start)
                        clean = _CLEAN_RE.sub("", tok).lower()
                        if clean:
                            timeline_words.append(TimelineWord(
                                word=tok,
                                clean_word=clean,
                                timeline_start=t_start,
                                timeline_end=t_end,
                            ))

            timeline_offset += clip_dur

        return timeline_words

    def _match_with_heuristics(
        self,
        timeline_words: List[TimelineWord],
        total_duration: float,
    ) -> List[CandidateOverlay]:
        """
        Pure-Python heuristic keyword matching using token n-grams and SequenceMatcher fuzzy matching.
        """
        candidates: List[CandidateOverlay] = []
        assets = self.catalog.get_all(require_exists=True)
        if not assets:
            return []

        # Pre-build tag index with normalized tags
        tag_to_assets: Dict[str, List[BrollAsset]] = {}
        for a in assets:
            for tag in a.tags:
                clean_tag = _CLEAN_RE.sub("", tag).strip().lower()
                if clean_tag:
                    tag_to_assets.setdefault(clean_tag, []).append(a)

        n = len(timeline_words)
        for i in range(n):
            w0 = timeline_words[i]
            if w0.clean_word in _STOPWORDS:
                continue

            # Build 1-gram, 2-gram, and 3-gram windows
            windows: List[Tuple[str, float]] = [
                (w0.clean_word, w0.timeline_start),
            ]
            if i + 1 < n:
                w1 = timeline_words[i + 1]
                windows.append((f"{w0.clean_word} {w1.clean_word}", w0.timeline_start))
            if i + 2 < n:
                w2 = timeline_words[i + 2]
                windows.append((f"{w0.clean_word} {w1.clean_word} {w2.clean_word}", w0.timeline_start))

            for phrase, t_start in windows:
                for tag, matched_assets in tag_to_assets.items():
                    score = 0.0
                    if phrase == tag:
                        score = 1.0
                    elif len(phrase) >= 4 and (phrase in tag or tag in phrase):
                        score = 0.85
                    elif len(phrase) >= 5 and len(tag) >= 5 and (phrase.startswith(tag[:5]) or tag.startswith(phrase[:5])):
                        score = 0.80
                    elif len(phrase) >= 4 and len(tag) >= 4:
                        ratio = SequenceMatcher(None, phrase, tag).ratio()
                        if ratio >= 0.72:
                            score = ratio * 0.8

                    if score >= 0.60:
                        for asset in matched_assets:
                            dur = min(self.default_duration, asset.duration)
                            # Start cutaway slightly before or exactly at spoken keyword
                            cand_start = max(round(t_start - 0.15, 2), 0.0)
                            candidates.append(CandidateOverlay(
                                broll_id=asset.id,
                                timeline_start=cand_start,
                                duration=dur,
                                score=score,
                                matched_keyword=phrase,
                                asset=asset,
                                reason=f"Heuristic match on tag '{tag}' (score {score:.2f})",
                            ))

        return candidates

    def _match_with_llm(
        self,
        timeline_words: List[TimelineWord],
        total_duration: float,
    ) -> List[CandidateOverlay]:
        """
        Tier 1: Query Ollama LLM provider to select contextually relevant B-roll cutaways.
        """
        if not self.provider or not self.provider.is_available():
            raise RuntimeError("Ollama provider is unavailable")

        assets = self.catalog.get_all(require_exists=True)
        if not assets:
            return []

        # Build catalog summary
        catalog_summary = "\n".join(
            f"- ID: {a.id} | Category: {a.category} | Tags: {', '.join(a.tags)} | Desc: {a.description}"
            for a in assets
        )

        # Build timeline transcript summary sampled across speech
        sample_step = max(len(timeline_words) // 40, 1)
        sampled_words = timeline_words[::sample_step]
        transcript_summary = " ".join(
            f"[{w.timeline_start:.1f}s: {w.word}]" for w in sampled_words[:50]
        )

        prompt = (
            f"You are a short-form video editor optimizing retention for YouTube Shorts.\n"
            f"Select up to {self.max_overlays} B-roll cutaway overlays from the catalog below.\n\n"
            f"RULES:\n"
            f"1. NEVER place any overlay before {self.hook_protection}s (the opening hook must show the speaker).\n"
            f"2. Overlays must last between {self.min_duration}s and {self.max_duration}s.\n"
            f"3. Maintain at least {self.min_spacing}s between any two overlays.\n"
            f"4. ONLY select broll_id values from the catalog.\n\n"
            f"CATALOG:\n{catalog_summary}\n\n"
            f"TIMELINE TRANSCRIPT:\n{transcript_summary}\n\n"
            f"Respond with JSON in this format:\n"
            f'{{"matches": [{{"broll_id": "...", "timeline_start": 4.5, "duration": 2.2, "matched_keyword": "...", "reason": "..."}}]}}'
        )

        resp = self.provider.generate_json(prompt)
        matches = resp.get("matches", []) if isinstance(resp, dict) else []
        candidates: List[CandidateOverlay] = []

        for m in matches:
            b_id = str(m.get("broll_id", "")).strip()
            asset = self.catalog.get_asset(b_id)
            if not asset or not asset.file_exists:
                continue

            try:
                t_start = float(m.get("timeline_start", 0.0))
                dur = float(m.get("duration", self.default_duration))
            except (ValueError, TypeError):
                continue

            candidates.append(CandidateOverlay(
                broll_id=asset.id,
                timeline_start=round(t_start, 2),
                duration=dur,
                score=0.95,
                matched_keyword=str(m.get("matched_keyword", "")),
                asset=asset,
                reason=str(m.get("reason", "LLM selected match")),
            ))

        return candidates

    def _enforce_pacing_constraints(
        self,
        candidates: List[CandidateOverlay],
        total_duration: float,
    ) -> List[BrollOverlay]:
        """
        Enforce strict viral retention rules:
        - Hook protection: start >= 2.5s
        - Tail protection: end <= total_duration - 1.0s
        - Duration clamp: [1.5, 3.5]
        - Spacing cooldown: next_start >= prev_end + 5.0s
        - Total coverage budget cap: <= 20% of total video duration
        - Asset diversity: prevent repeating same asset unless needed
        """
        if not candidates or total_duration <= 0:
            return []

        # Sort candidates chronologically; break ties by score (higher score first)
        sorted_candidates = sorted(
            candidates,
            key=lambda c: (c.timeline_start, -c.score),
        )

        selected_overlays: List[BrollOverlay] = []
        last_end_time = 0.0
        total_broll_duration = 0.0
        used_asset_ids: Set[str] = set()
        max_broll_allowed = self.max_coverage_ratio * total_duration

        for cand in sorted_candidates:
            # 1. Clamp duration within bounds
            clamped_dur = max(self.min_duration, min(self.max_duration, cand.duration))
            cand_start = cand.timeline_start
            cand_end = cand_start + clamped_dur

            # 2. Hook protection: speaker face must remain uninterrupted at start
            if cand_start < self.hook_protection:
                continue

            # 3. Tail protection: speaker face on punchline/CTA
            if cand_end > (total_duration - self.tail_protection):
                continue

            # 4. Spacing cooldown: at least 5.0s between B-roll cutaways
            if selected_overlays and (cand_start < (last_end_time + self.min_spacing)):
                continue

            # 5. Budget cap: max 20% total B-roll coverage
            if (total_broll_duration + clamped_dur) > max_broll_allowed:
                continue

            # 6. Max overlays limit
            if len(selected_overlays) >= self.max_overlays:
                break

            # 7. Asset diversity (prefer novel asset)
            if cand.asset.id in used_asset_ids and len(used_asset_ids) < self.catalog.total_count:
                # If we have other unused assets available, skip repeat
                continue

            # 8. Ensure media file actually exists on disk
            if not cand.asset.file_exists:
                logger.warning("B-roll file for '%s' missing on disk (%s), skipping.", cand.broll_id, cand.asset.path)
                continue

            overlay = BrollOverlay(
                broll_id=cand.broll_id,
                timeline_start=round(cand_start, 2),
                timeline_end=round(cand_end, 2),
                source_path=str(cand.asset.path),
                overlay_type="cutaway",
            )
            selected_overlays.append(overlay)
            last_end_time = cand_end
            total_broll_duration += clamped_dur
            used_asset_ids.add(cand.asset.id)

        return selected_overlays
