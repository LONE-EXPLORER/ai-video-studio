"""
app/agents/editorial_agent.py — LLM-based editorial intelligence.

The EditorialAgent analyses a transcript and video metadata, then uses an LLM
provider to produce a structured EditingPlan. It validates the plan against the
transcript and retries on validation failure.
"""

from __future__ import annotations

import json
import uuid
from typing import Callable, List, Optional

from app.providers.base import BaseProvider
from app.schemas.editing_plan import AspectRatio, EditingPlan
from app.schemas.transcript import Transcript
from app.schemas.video_meta import VideoMetadata
from app.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a world-class professional video editor with 20 years of experience \
creating viral social media content, documentary films, and cinematic shorts.

Your task is to analyse a provided video transcript and metadata, then produce \
a precise, professional editing plan in JSON format.

## Fundamental rules you MUST follow:
1. ALL clip timestamps (source_start, source_end) MUST be real timestamps that \
exist within the transcript. Do NOT invent timestamps.
2. Do NOT fabricate dialogue or attribute words to speakers if they did not say them.
3. The sum of all clip durations should approximate the requested target duration.
4. Every clip must have a clear creative reason for its inclusion.
5. The hook (first clip) must be the most compelling, attention-grabbing moment.
6. Output ONLY valid JSON. No markdown. No commentary before or after the JSON object.

## Output schema:
{
  "plan_id": "<uuid string>",
  "job_id": "<provided job_id>",
  "profile": "<dynamic_shorts|long_form|cinematic>",
  "aspect_ratio": "<9:16|16:9|1:1>",
  "target_duration_seconds": <float>,
  "clips": [
    {
      "clip_id": "<short unique id e.g. c001>",
      "source_start": <float seconds from transcript>,
      "source_end": <float seconds from transcript>,
      "order": <integer starting at 1>,
      "reason": "<why this clip is included>",
      "zoom": {"enabled": false, "factor": 1.05},
      "transition_in": "cut",
      "transition_out": "cut"
    }
  ],
  "caption_settings": {
    "enabled": true,
    "style": "dynamic",
    "font_size": 52,
    "position": "bottom",
    "word_highlight_color": "#FFD700",
    "text_color": "#FFFFFF",
    "outline_color": "#000000",
    "margin_bottom": 80
  },
  "audio_settings": {
    "normalize": true,
    "fade_in": 0.3,
    "fade_out": 0.5,
    "background_music_path": null,
    "music_volume": 0.15,
    "duck_music": true
  },
  "hook_summary": "<one sentence describing the opening hook>",
  "edit_rationale": "<paragraph explaining your editorial choices>"
}
"""


class EditorialAgent:
    """
    Uses an LLM provider to transform a transcript and metadata into a validated
    :class:`~app.schemas.editing_plan.EditingPlan`.

    Parameters
    ----------
    provider:
        Any concrete implementation of :class:`~app.providers.base.BaseProvider`.
    """

    def __init__(self, provider: BaseProvider) -> None:
        self.provider = provider

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_and_plan(
        self,
        transcript: Transcript,
        video_metadata: VideoMetadata,
        user_instruction: str,
        profile: str,
        target_duration: float,
        aspect_ratio: str,
        job_id: str,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> EditingPlan:
        """
        Produce a validated :class:`EditingPlan` for the given inputs.

        The agent builds a detailed prompt, calls the LLM, validates the result
        against the Pydantic schema and transcript, and retries once on failure.

        Parameters
        ----------
        transcript:
            Full speech transcript with word-level timestamps.
        video_metadata:
            Probed video metadata (resolution, duration, etc.).
        user_instruction:
            Free-text editing instruction from the user.
        profile:
            Editing profile (``"dynamic_shorts"``, ``"long_form"``, ``"cinematic"``).
        target_duration:
            Target output duration in seconds.
        aspect_ratio:
            Target aspect ratio string (``"9:16"``, ``"16:9"``, ``"1:1"``).
        job_id:
            Job UUID string to embed in the plan.
        log_callback:
            Optional callback invoked with human-readable sub-step log strings.

        Returns
        -------
        EditingPlan
            Validated editing plan.

        Raises
        ------
        RuntimeError
            If the LLM fails to produce a valid plan after one retry.
        """
        def _step_log(msg: str) -> None:
            logger.info(msg)
            if log_callback:
                try:
                    log_callback(msg)
                except Exception:
                    pass

        _step_log(
            f"Constructing editorial prompt for {self.provider.model_name} "
            f"({len(transcript.segments)} segments, target: {target_duration:.0f}s, {aspect_ratio})..."
        )

        prompt = self._build_prompt(
            transcript=transcript,
            video_metadata=video_metadata,
            user_instruction=user_instruction,
            profile=profile,
            target_duration=target_duration,
            aspect_ratio=aspect_ratio,
            job_id=job_id,
        )

        _step_log(
            f"Dispatching plan request to LLM ({self.provider.model_name}). "
            f"Generating structured JSON cut decision list..."
        )

        # First attempt
        first_error_msg = ""
        try:
            raw = self.provider.generate_json(prompt, system=_SYSTEM_PROMPT)
            _step_log("Received raw LLM response. Validating JSON schema and cut timestamps...")
            plan = self._parse_and_validate(raw, transcript, video_metadata)
            _step_log(
                f"Editing plan successfully validated! {len(plan.clips)} clips planned "
                f"(total: {plan.total_duration:.1f}s). Hook: \"{plan.hook_summary[:60]}\""
            )
            return plan
        except Exception as exc:
            first_error_msg = str(exc)
            _step_log(
                f"First plan generation attempt encountered an issue: {first_error_msg[:140]}... "
                f"Preparing automated recovery retry..."
            )

        # If primary timed out or failed and fallback is available, switch to fallback model
        if hasattr(self.provider, "_fallback_model") and self.provider._fallback_model:
            fallback = self.provider._fallback_model
            if fallback and fallback != self.provider.model_name:
                _step_log(f"Attempting failover to secondary model: '{fallback}'...")
                self.provider._active_model = fallback

        # Retry with error context
        retry_prompt = (
            f"{prompt}\n\n"
            f"## IMPORTANT — Your previous response had the following errors:\n"
            f"{first_error_msg}\n\n"
            f"Fix all issues and return valid JSON only. "
            f"Ensure source_start and source_end reference actual transcript timestamps. "
            f"Ensure source_end > source_start for every clip."
        )
        try:
            _step_log(f"Executing retry attempt with model {self.provider.model_name}...")
            raw = self.provider.generate_json(retry_prompt, system=_SYSTEM_PROMPT)
            _step_log("Received retry LLM response. Validating JSON schema and timestamps...")
            plan = self._parse_and_validate(raw, transcript, video_metadata)
            _step_log(
                f"Editing plan successfully recovered on retry! {len(plan.clips)} clips planned "
                f"(total: {plan.total_duration:.1f}s)."
            )
            return plan
        except Exception as second_error:
            _step_log(f"Retry plan generation failed: {second_error}")
            raise RuntimeError(
                f"Editorial agent failed to produce a valid plan after retry.\n"
                f"Last error: {second_error}"
            ) from second_error

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        transcript: Transcript,
        video_metadata: VideoMetadata,
        user_instruction: str,
        profile: str,
        target_duration: float,
        aspect_ratio: str,
        job_id: str,
    ) -> str:
        """Build the full prompt string to send to the LLM."""
        transcript_summary = self._build_transcript_summary(transcript)

        profile_guidance = {
            "dynamic_shorts": (
                "Create a punchy, fast-paced edit optimised for TikTok/Instagram Reels. "
                "Lead with the hook immediately. Keep clips short (3–8 seconds). "
                "Prioritise high-energy moments, surprising statements, and emotional peaks."
            ),
            "long_form": (
                "Create a coherent, narrative-driven edit for YouTube or podcast video. "
                "Maintain a logical story arc: introduction, development, conclusion. "
                "Remove filler words and dead air while preserving complete thoughts."
            ),
            "cinematic": (
                "Create a visually-driven, contemplative edit with breathing room. "
                "Allow moments to land. Prioritise visual storytelling and atmosphere. "
                "Use fades and longer clips to create an emotional experience."
            ),
        }.get(profile, "Create a well-paced edit matching the user's instruction.")

        return f"""
## VIDEO INFORMATION

- Job ID: {job_id}
- Source duration: {video_metadata.duration_seconds:.1f} seconds
- Resolution: {video_metadata.video_stream.width}x{video_metadata.video_stream.height}
- Frame rate: {video_metadata.frame_rate:.2f} fps
- Has audio: {video_metadata.has_audio}
- Detected language: {transcript.language}

## EDITING PROFILE: {profile.upper()}

{profile_guidance}

## USER INSTRUCTION

"{user_instruction}"

## TARGET OUTPUT

- Duration: {target_duration:.0f} seconds (± 15% acceptable)
- Aspect ratio: {aspect_ratio}
- Plan profile field: "{profile}"
- Plan aspect_ratio field: "{aspect_ratio}"

## FULL TRANSCRIPT WITH TIMESTAMPS

{transcript_summary}

## YOUR TASK

Analyse the transcript above and produce a JSON editing plan.
- Select clips whose combined duration approximates {target_duration:.0f} seconds.
- ALL source_start and source_end values MUST be timestamps from the transcript above.
- Do NOT use timestamps beyond {video_metadata.duration_seconds:.1f} seconds.
- Order clips by narrative logic, not chronological order if reordering serves the story.
- Assign the plan_id field a new UUID (generate one yourself).
- Set job_id to: "{job_id}"

Return ONLY the JSON object. No explanation text before or after.
""".strip()

    def _build_transcript_summary(self, transcript: Transcript) -> str:
        """
        Format the transcript with timestamps for embedding in the LLM prompt.

        Returns a compact representation: ``[start→end] text``
        """
        lines = []
        for seg in transcript.segments:
            lines.append(
                f"[{seg.start:.2f}s → {seg.end:.2f}s] {seg.text.strip()}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _parse_and_validate(
        self,
        raw: dict,
        transcript: Transcript,
        video_metadata: VideoMetadata,
    ) -> EditingPlan:
        """
        Parse the LLM response dict into an EditingPlan and validate it.

        Parameters
        ----------
        raw:
            Parsed JSON dict from the LLM.
        transcript:
            Source transcript for timestamp validation.
        video_metadata:
            Source video metadata for duration validation.

        Returns
        -------
        EditingPlan

        Raises
        ------
        ValueError
            If Pydantic validation fails or timestamps are invalid.
        """
        # Pydantic validation
        plan = EditingPlan(**raw)

        # Semantic validation
        errors = self._validate_plan_against_transcript(plan, transcript, video_metadata)
        # Also validate against source duration
        ts_errors = plan.validate_timestamps(video_metadata.duration_seconds)
        all_errors = errors + ts_errors

        if all_errors:
            raise ValueError(
                "Plan timestamp validation failed:\n" + "\n".join(all_errors)
            )

        return plan

    def _validate_plan_against_transcript(
        self,
        plan: EditingPlan,
        transcript: Transcript,
        video_metadata: Optional[VideoMetadata] = None,
    ) -> List[str]:
        """
        Check that all clip timestamps fall within the transcript or video duration range.
        """
        errors: List[str] = []
        max_duration = transcript.duration
        if video_metadata is not None:
            max_duration = max(max_duration, video_metadata.duration_seconds)

        for clip in plan.clips:
            if clip.source_start < 0:
                errors.append(
                    f"Clip {clip.clip_id}: source_start {clip.source_start} < 0"
                )
            if clip.source_end > max_duration + 1.0:  # 1-second tolerance
                errors.append(
                    f"Clip {clip.clip_id}: source_end {clip.source_end:.2f} "
                    f"exceeds duration {max_duration:.2f}s"
                )
            if clip.source_end <= clip.source_start:
                errors.append(
                    f"Clip {clip.clip_id}: source_end {clip.source_end:.2f} "
                    f"<= source_start {clip.source_start:.2f}"
                )

        if not plan.clips:
            errors.append("Plan has no clips")

        return errors
