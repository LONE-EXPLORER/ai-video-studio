"""
app/rendering/renderer.py — Final render pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from app.captions.caption_engine import CaptionEngine
from app.editing.audio_processor import duck_music
from app.editing.editor import VideoEditor, _ASPECT_RATIO_DIMS
from app.schemas.editing_plan import EditingPlan
from app.utils import ffmpeg_utils
from app.utils.logger import get_logger

logger = get_logger(__name__)


class Renderer:
    """
    Applies final rendering steps: aspect ratio, captions, audio mix, H.264 export.
    """

    def __init__(self) -> None:
        self._editor = VideoEditor()
        self._caption_engine = CaptionEngine()

    def render_final_video(
        self,
        rough_cut: Path,
        plan: EditingPlan,
        job_dirs: Dict[str, Path],
        transcript=None,
        output_name: str = "final.mp4",
    ) -> Path:
        """
        Apply all final rendering steps and produce the export-ready MP4.

        Steps:
        1. Aspect ratio conversion.
        2. Burn captions (if enabled in the plan).
        3. Background music mixing (if configured).
        4. Final H.264/AAC render to exports directory.

        Parameters
        ----------
        rough_cut:
            Path to the rough cut video from the editing stage.
        plan:
            Validated editing plan.
        job_dirs:
            Dict mapping directory names to paths.
        transcript:
            Optional Transcript object for caption generation.
        output_name:
            Output filename (default ``"final.mp4"``).

        Returns
        -------
        Path
            Path to the final rendered export file.
        """
        intermediate_dir = job_dirs["intermediate"]
        exports_dir = job_dirs["exports"]
        exports_dir.mkdir(parents=True, exist_ok=True)

        current = rough_cut

        # 1. Aspect ratio
        width, height = _ASPECT_RATIO_DIMS.get(plan.aspect_ratio, (1920, 1080))
        ar_path = intermediate_dir / "ar_converted.mp4"
        logger.info("Applying aspect ratio %s → %dx%d", plan.aspect_ratio, width, height)
        ffmpeg_utils.render_final(current, ar_path, width=width, height=height)
        current = ar_path

        # 1.5. B-Roll Overlay Compositing (Feature 11)
        if plan.broll_overlays:
            from app.rendering.broll_compositor import BrollCompositor

            broll_compositor = BrollCompositor()
            broll_path = intermediate_dir / "broll_composited.mp4"
            logger.info("Compositing %d B-roll overlay(s) on conformed video...", len(plan.broll_overlays))
            try:
                current = broll_compositor.composite_overlays(
                    base_video_path=current,
                    overlays=plan.broll_overlays,
                    output_path=broll_path,
                    target_w=width,
                    target_h=height,
                )
            except Exception as exc:
                logger.warning("B-roll compositing failed (%s); proceeding with base video", exc)

        # 2. Captions
        caption_settings = plan.caption_settings
        if caption_settings.enabled and transcript is not None:
            ass_path = job_dirs["transcripts"] / "captions.ass"
            self._caption_engine.generate_ass_captions(
                transcript=transcript,
                style_name=caption_settings.style,
                output_path=ass_path,
                highlight_words=True,
            )
            try:
                from app.captions.pillow_captions import burn_kinetic_captions

                captioned_path = intermediate_dir / "captioned.mp4"
                logger.info("Burning kinetic typography with word-level highlight into video...")
                burn_kinetic_captions(
                    video_path=current,
                    plan=plan,
                    transcript=transcript,
                    output_path=captioned_path,
                    width=width,
                    height=height,
                    font_size=getattr(caption_settings, "font_size", 72) or 72,
                )
                if captioned_path.exists() and captioned_path.stat().st_size > 0:
                    current = captioned_path
                    logger.info("Successfully burned kinetic typography captions into video stream!")
            except Exception as exc:
                logger.warning(
                    "Pillow kinetic caption burn-in fallback: %s. Subtitle files remain available in exports.",
                    exc,
                )
        elif caption_settings.enabled:
            logger.warning("Captions enabled but no transcript provided; skipping.")

        # 3. SFX Mixing (Feature 15 — M5)
        sfx_events = []
        try:
            from app.editing.sfx_mixer import SFXMixer
            sfx_mixer = SFXMixer()
            sfx_events = sfx_mixer.build_events_from_plan(plan)
            if sfx_events:
                sfx_path = intermediate_dir / "sfx_mixed.mp4"
                result = sfx_mixer.mix_sfx(
                    dialogue_video=current,
                    events=sfx_events,
                    output_path=sfx_path,
                    overall_sfx_volume=0.68,
                )
                if result.success and sfx_path.exists() and sfx_path.stat().st_size > 0:
                    current = sfx_path
                    logger.info("SFX mixing complete: %d events placed", result.events_placed)
                else:
                    logger.warning("SFX mixing skipped (failed); proceeding without SFX")
        except Exception as exc:
            logger.warning("SFX mixing error (%s); proceeding without SFX", exc)

        # 4. Background music — Sidechain ducking (Feature 16 — M5)
        audio_settings = plan.audio_settings
        if audio_settings.background_music_path:
            music_path = Path(audio_settings.background_music_path)
            if music_path.exists():
                sidechain_path = intermediate_dir / "sidechain_mixed.mp4"
                try:
                    from app.editing.sidechain_ducker import SidechainDucker, DuckingParams
                    ducker = SidechainDucker(DuckingParams(
                        attack_ms=5.0,
                        release_ms=250.0,
                        threshold_db=-24.0,
                        ratio=8.0,
                        music_pre_vol=audio_settings.music_volume,
                    ))
                    result_path = ducker.duck_and_mix(
                        dialogue_video=current,
                        music_path=music_path,
                        output_path=sidechain_path,
                        music_volume=audio_settings.music_volume,
                    )
                    if sidechain_path.exists() and sidechain_path.stat().st_size > 0:
                        current = sidechain_path
                        logger.info("Sidechain ducking complete (vol=%.2f)", audio_settings.music_volume)
                    else:
                        raise RuntimeError("Sidechain output empty")
                except Exception as exc:
                    logger.warning("Sidechain ducking failed (%s); falling back to static duck_music", exc)
                    mixed_path = intermediate_dir / "music_mixed.mp4"
                    duck_music(
                        dialogue_path=current,
                        music_path=music_path,
                        output_path=mixed_path,
                        music_vol=audio_settings.music_volume,
                    )
                    if mixed_path.exists() and mixed_path.stat().st_size > 0:
                        current = mixed_path
                        logger.info("Static music mix complete (vol=%.2f)", audio_settings.music_volume)
            else:
                logger.warning(
                    "Background music file not found: %s — skipping",
                    music_path,
                )

        # 5. Final render
        output_path = exports_dir / output_name
        ffmpeg_utils.render_final(current, output_path, width=width, height=height)
        logger.info("Final render complete → %s", output_path)
        self.last_sfx_events = sfx_events
        return output_path
