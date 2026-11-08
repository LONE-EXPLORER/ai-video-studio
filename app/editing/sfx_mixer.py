"""
app/editing/sfx_mixer.py — Sound effect placement and mixing engine.

Places whoosh/impact/riser SFX at precise timecodes using FFmpeg adelay,
then merges them with the main dialogue/music track.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from app.schemas.editing_plan import EditingPlan
from app.utils.logger import get_logger

logger = get_logger(__name__)

FFMPEG = "/opt/homebrew/bin/ffmpeg"
_SFX_INDEX_PATH = Path("assets/sfx/sfx_index.json")


@dataclass
class SFXEvent:
    """A single SFX placement event."""
    sfx_name: str          # key in sfx_index.json
    timecode: float        # seconds from start of video where SFX begins
    volume: float = 1.0   # multiplier (0.0–1.0+)
    pan: float = 0.0       # stereo pan -1.0 (left) to 1.0 (right)


@dataclass
class SFXMixResult:
    output_path: Path
    events_placed: int
    success: bool
    error: Optional[str] = None


class SFXMixer:
    """
    Places SFX events at precise timecodes relative to cut list from EditingPlan.

    Strategy:
    - whoosh.wav  → placed 20ms before every jump cut (punch-in)
    - impact.wav  → placed at the hook word (first cut, 0s)
    - swoosh_short.wav → placed at every Nth alternating cut for variety
    - riser.wav   → placed 2s before the final cut if video > 30s
    """

    def __init__(self, sfx_root: Path = Path("assets/sfx")) -> None:
        self._sfx_root = sfx_root
        self._index = self._load_index()

    def _load_index(self) -> dict:
        idx_path = self._sfx_root / "sfx_index.json"
        if idx_path.exists():
            return json.loads(idx_path.read_text())
        logger.warning("SFX index not found at %s; using empty index", idx_path)
        return {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_events_from_plan(self, plan: EditingPlan) -> List[SFXEvent]:
        """
        Derive SFX event list from the EditingPlan's clip cut points.
        Returns list of SFXEvent sorted by timecode.
        """
        if not plan.clips:
            return []

        events: List[SFXEvent] = []
        sorted_clips = sorted(plan.clips, key=lambda c: c.order)

        # Timeline of cut points (absolute timecodes of each clip start)
        cut_timecodes: List[float] = []
        cursor = 0.0
        for clip in sorted_clips:
            cut_timecodes.append(cursor)
            cursor += clip.duration

        # 1. Impact boom at hook (t=0)
        if "impact" in self._index:
            events.append(SFXEvent(
                sfx_name="impact",
                timecode=0.0,
                volume=0.75,
            ))

        # 2. Whoosh 50ms before each punch-in cut (order ≥ 2)
        whoosh_vol = self._index.get("whoosh", {}).get("volume", 0.6)
        swoosh_vol = self._index.get("swoosh_short", {}).get("volume", 0.5)
        for i, tc in enumerate(cut_timecodes[1:], start=1):
            # Alternate between whoosh and swoosh_short for variety
            if i % 3 == 0 and "swoosh_short" in self._index:
                events.append(SFXEvent(
                    sfx_name="swoosh_short",
                    timecode=max(0.0, tc - 0.05),
                    volume=swoosh_vol * 0.9,
                ))
            elif "whoosh" in self._index:
                events.append(SFXEvent(
                    sfx_name="whoosh",
                    timecode=max(0.0, tc - 0.05),
                    volume=whoosh_vol,
                ))

        # 3. Riser 2s before the last 20% of the video
        total_dur = cursor
        if total_dur > 25.0 and "riser" in self._index:
            riser_tc = max(0.0, total_dur * 0.78 - 2.0)
            events.append(SFXEvent(
                sfx_name="riser",
                timecode=riser_tc,
                volume=self._index["riser"].get("volume", 0.4),
            ))

        events.sort(key=lambda e: e.timecode)
        logger.info("Built %d SFX events from %d clips", len(events), len(sorted_clips))
        return events

    def mix_sfx(
        self,
        dialogue_video: Path,
        events: List[SFXEvent],
        output_path: Path,
        overall_sfx_volume: float = 0.70,
    ) -> SFXMixResult:
        """
        Mix SFX events into the video's audio track using FFmpeg adelay.

        Uses amix to blend SFX with original dialogue audio.
        """
        if not events:
            logger.info("No SFX events to mix; copying video as-is")
            import shutil
            shutil.copy2(dialogue_video, output_path)
            return SFXMixResult(output_path=output_path, events_placed=0, success=True)

        # Build FFmpeg filter_complex with adelay per SFX
        # Input 0 = dialogue video, inputs 1..N = SFX files
        filter_parts: List[str] = []
        amix_inputs: List[str] = ["[0:a]"]

        valid_events: List[SFXEvent] = []
        for ev in events:
            sfx_info = self._index.get(ev.sfx_name)
            if sfx_info is None:
                logger.warning("SFX '%s' not in index; skipping", ev.sfx_name)
                continue
            sfx_path = Path(sfx_info["path"])
            if not sfx_path.is_absolute():
                sfx_path = Path("assets/sfx") / sfx_path.name
            if not sfx_path.exists():
                logger.warning("SFX file not found: %s; skipping", sfx_path)
                continue
            valid_events.append(ev)

        if not valid_events:
            import shutil
            shutil.copy2(dialogue_video, output_path)
            return SFXMixResult(output_path=output_path, events_placed=0, success=True)

        cmd = [FFMPEG, "-y", "-i", str(dialogue_video)]

        # Add each SFX as separate input
        for ev in valid_events:
            sfx_info = self._index[ev.sfx_name]
            sfx_path = Path("assets/sfx") / Path(sfx_info["path"]).name
            cmd.extend(["-i", str(sfx_path)])

        # Build filter_complex
        for i, ev in enumerate(valid_events, start=1):
            delay_ms = int(ev.timecode * 1000)
            vol = ev.volume * overall_sfx_volume
            label = f"sfx{i}"
            filter_parts.append(
                f"[{i}:a]adelay={delay_ms}|{delay_ms},"
                f"volume={vol:.3f},"
                f"apad=pad_dur=0[{label}]"
            )
            amix_inputs.append(f"[{label}]")

        n_inputs = len(amix_inputs)
        amix_str = "".join(amix_inputs) + f"amix=inputs={n_inputs}:duration=first:normalize=0[amixed]"
        filter_complex = ";".join(filter_parts) + (";" if filter_parts else "") + amix_str

        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "0:v",
            "-map", "[amixed]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "48000",
            "-ac", "2",
            str(output_path),
        ])

        logger.info(
            "Mixing %d SFX events into audio track (overall_vol=%.2f)...",
            len(valid_events), overall_sfx_volume
        )

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                logger.error("SFX mix FFmpeg failed:\n%s", result.stderr[-600:])
                # Fallback: copy original
                import shutil
                shutil.copy2(dialogue_video, output_path)
                return SFXMixResult(
                    output_path=output_path,
                    events_placed=0,
                    success=False,
                    error=result.stderr[-300:],
                )
            logger.info("SFX mix complete → %s (%d events placed)", output_path, len(valid_events))
            return SFXMixResult(
                output_path=output_path,
                events_placed=len(valid_events),
                success=True,
            )
        except subprocess.TimeoutExpired:
            logger.error("SFX mix timed out")
            import shutil
            shutil.copy2(dialogue_video, output_path)
            return SFXMixResult(output_path=output_path, events_placed=0, success=False, error="timeout")
