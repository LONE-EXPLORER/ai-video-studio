"""
app/editing/silence_remover.py — Margin-padded silence detection and speech segment extraction.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from app.utils.ffmpeg_utils import find_ffmpeg
from app.utils.logger import get_logger

logger = get_logger(__name__)


def detect_silence(
    audio_path: Path,
    min_silence_duration: float = 0.5,
    silence_threshold_db: float = -35.0,
    duration: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """
    Detect silent regions in an audio file using FFmpeg's ``silencedetect`` filter.
    Uses stateful stream parsing to handle unclosed EOF silences, negative timestamps,
    and missing start tags.

    Parameters
    ----------
    audio_path:
        Path to the audio or video file.
    min_silence_duration:
        Minimum duration in seconds for a region to be classified as silence.
    silence_threshold_db:
        Noise floor threshold in dBFS (default -35.0 dB).
    duration:
        Optional known total duration to bound unclosed trailing silences.

    Returns
    -------
    List[Tuple[float, float]]
        List of ``(start_seconds, end_seconds)`` tuples for each silent region.
    """
    ffmpeg = find_ffmpeg()
    args = [
        ffmpeg,
        "-i", str(audio_path),
        "-af", (
            f"silencedetect=noise={silence_threshold_db}dB:"
            f"d={min_silence_duration}"
        ),
        "-f", "null",
        "-",
    ]

    logger.debug(
        "Detecting silence: threshold=%.1f dB, min_duration=%.2f s",
        silence_threshold_db,
        min_silence_duration,
    )
    result = subprocess.run(args, capture_output=True, text=True, timeout=300)
    stderr = result.stderr

    silences: List[Tuple[float, float]] = []
    current_start: Optional[float] = None

    # Stateful line-by-line parsing
    for line in stderr.splitlines():
        start_match = re.search(r"silence_start:\s*(-?[\d.]+(?:e-?\d+)?)", line)
        if start_match:
            current_start = max(0.0, float(start_match.group(1)))

        end_match = re.search(r"silence_end:\s*(-?[\d.]+(?:e-?\d+)?)", line)
        if end_match:
            end_val = max(0.0, float(end_match.group(1)))
            if current_start is not None:
                start_val = current_start
                current_start = None
            else:
                # Silence started at 0.0 but FFmpeg omitted silence_start
                dur_match = re.search(r"silence_duration:\s*(-?[\d.]+(?:e-?\d+)?)", line)
                if dur_match:
                    start_val = max(0.0, end_val - float(dur_match.group(1)))
                else:
                    start_val = 0.0

            if end_val > start_val:
                silences.append((round(start_val, 4), round(end_val, 4)))

    # If stream ended while silence was still ongoing
    if current_start is not None and duration is not None:
        if duration > current_start:
            silences.append((round(current_start, 4), round(duration, 4)))

    logger.info("Detected %d silent regions in %s", len(silences), audio_path.name)
    return silences


def merge_overlapping_segments(
    segments: List[Tuple[float, float]],
    epsilon: float = 1e-4,
) -> List[Tuple[float, float]]:
    """
    Merge overlapping or contiguous time intervals into unified, sorted intervals.

    Parameters
    ----------
    segments:
        List of ``(start, end)`` tuples.
    epsilon:
        Tolerance for considering adjacent intervals contiguous.

    Returns
    -------
    List[Tuple[float, float]]
        Sorted list of non-overlapping ``(start, end)`` intervals.
    """
    if not segments:
        return []

    sorted_segs = sorted(segments, key=lambda s: (s[0], s[1]))
    merged: List[List[float]] = []

    for start, end in sorted_segs:
        if start >= end:
            continue

        if not merged:
            merged.append([start, end])
        else:
            prev_start, prev_end = merged[-1]
            if start <= prev_end + epsilon:
                merged[-1][1] = max(prev_end, end)
            else:
                merged.append([start, end])

    return [(round(s, 4), round(e, 4)) for s, e in merged]


def apply_margin_padding(
    segments: List[Tuple[float, float]],
    source_duration: float,
    lead_margin: float = 0.080,
    tail_margin: float = 0.180,
    merge: bool = True,
) -> List[Tuple[float, float]]:
    """
    Expand speech intervals with lead-in and tail-out margins to preserve
    plosive consonants and breath decays, clamping to [0, source_duration].

    Parameters
    ----------
    segments:
        List of raw ``(start, end)`` speech segment tuples.
    source_duration:
        Total duration of the source media in seconds.
    lead_margin:
        Seconds to expand speech segment backward (default 0.080s / 80ms).
    tail_margin:
        Seconds to expand speech segment forward (default 0.180s / 180ms).
    merge:
        If True, merges resulting overlapping segments into unified intervals.

    Returns
    -------
    List[Tuple[float, float]]
        Padded (and optionally merged) speech segment intervals.
    """
    if not segments or source_duration <= 0.0:
        return []

    padded: List[Tuple[float, float]] = []
    for start, end in segments:
        new_start = max(0.0, start - lead_margin)
        new_end = min(source_duration, end + tail_margin)
        if new_end > new_start:
            padded.append((round(new_start, 4), round(new_end, 4)))

    if merge:
        return merge_overlapping_segments(padded)
    return padded


def get_speech_segments(
    audio_path: Path,
    duration: float,
    min_silence_duration: float = 0.5,
    silence_threshold_db: float = -35.0,
    lead_margin: float = 0.080,
    tail_margin: float = 0.180,
    min_speech_duration: float = 0.0,
) -> List[Tuple[float, float]]:
    """
    Return inverted silence regions with +80ms lead and +180ms tail margin padding,
    merged cleanly across sub-margin pauses.

    Parameters
    ----------
    audio_path:
        Path to the audio file.
    duration:
        Total audio duration in seconds.
    min_silence_duration:
        Passed to :func:`detect_silence`.
    silence_threshold_db:
        Passed to :func:`detect_silence`.
    lead_margin:
        Lead-in padding (default 0.080s).
    tail_margin:
        Tail-out padding (default 0.180s).
    min_speech_duration:
        Optional minimum speech length; shorter segments are discarded.

    Returns
    -------
    List[Tuple[float, float]]
        List of ``(start, end)`` tuples for clean speech regions.
    """
    if duration <= 0.0:
        return []

    silences = detect_silence(
        audio_path,
        min_silence_duration=min_silence_duration,
        silence_threshold_db=silence_threshold_db,
        duration=duration,
    )

    raw_speech: List[Tuple[float, float]] = []
    cursor = 0.0

    for silence_start, silence_end in sorted(silences):
        if cursor < silence_start:
            raw_speech.append((cursor, silence_start))
        cursor = max(cursor, silence_end)

    if cursor < duration:
        raw_speech.append((cursor, duration))

    # Apply margin padding and overlap merging
    padded_speech = apply_margin_padding(
        raw_speech,
        source_duration=duration,
        lead_margin=lead_margin,
        tail_margin=tail_margin,
        merge=True,
    )

    if min_speech_duration > 0.0:
        padded_speech = [
            (s, e) for s, e in padded_speech if (e - s) >= min_speech_duration
        ]

    logger.debug("Identified %d padded speech segments", len(padded_speech))
    return padded_speech

