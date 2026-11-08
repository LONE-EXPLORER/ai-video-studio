"""
app/pipeline/orchestrator.py — Main pipeline orchestrator that coordinates all stages.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from app.agents.editorial_agent import EditorialAgent
from app.analysis.frame_extractor import extract_keyframes, generate_contact_sheet
from app.analysis.visual_analyzer import VisualAnalyzer
from app.config import get_config
from app.editing.editor import VideoEditor
from app.export.fcpxml_generator import generate_fcpxml
from app.export.fcpxml_multitrack import generate_multitrack_fcpxml
from app.pipeline.stage_runner import StageRunner, run_stages
from app.providers.base import BaseProvider
from app.quality.checker import QualityChecker
from app.rendering.preview_gen import generate_preview
from app.rendering.renderer import Renderer
import uuid
from app.schemas.editing_plan import AspectRatio, EditingPlan, Clip, ZoomEffect, CaptionSettings, AudioSettings
from app.schemas.transcript import Transcript
from app.schemas.video_meta import VideoMetadata
from app.transcription.srt_generator import generate_ass, generate_srt
from app.transcription.whisper_engine import TranscriptionEngine
from app.utils import db, ffmpeg_utils, file_utils
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PipelineOrchestrator:
    """
    Coordinates end-to-end processing of a video editing job through all pipeline stages.
    """

    def __init__(self, provider: BaseProvider) -> None:
        self.provider = provider
        self.agent = EditorialAgent(provider)
        self.config = get_config()
        self.editor = VideoEditor()
        self.renderer = Renderer()
        self.quality_checker = QualityChecker()
        self.analyzer = VisualAnalyzer()

    def run_job(
        self,
        job_id: str,
        source_video_path: Optional[Path] = None,
        user_instruction: str = "",
        profile: str = "dynamic_shorts",
        target_duration: float = 45.0,
        aspect_ratio: str = "9:16",
        caption_style: str = "dynamic",
        enable_music: bool = False,
        jobs_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[str, str, str], None]] = None,
        source_video_paths: Optional[List[Path]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full video editing pipeline for a given job.

        Parameters
        ----------
        job_id:
            Unique identifier for this job.
        source_video_path:
            Original primary video path on disk (or first source if multi-video).
        user_instruction:
            Natural language instruction from the user.
        profile:
            Editing profile: 'dynamic_shorts', 'long_form', or 'cinematic'.
        target_duration:
            Desired final video duration in seconds.
        aspect_ratio:
            Target aspect ratio ('9:16', '16:9', '1:1').
        caption_style:
            Caption presentation style ('dynamic', 'clean', 'minimal').
        enable_music:
            Whether background music should be applied.
        jobs_dir:
            Top-level directory to store job assets. Defaults to config.jobs_dir.
        progress_callback:
            Callback (stage_name, status, message) for UI progress updates.
        source_video_paths:
            Optional list of source video paths for multi-video workflows.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing output paths, plans, transcript, and quality report.
        """
        if jobs_dir is None:
            jobs_dir = self.config.jobs_dir

        if source_video_paths is None:
            if source_video_path is not None:
                source_video_paths = [Path(source_video_path)]
            else:
                raise ValueError("Either source_video_path or source_video_paths must be provided.")
        else:
            source_video_paths = [Path(p) for p in source_video_paths]
            if not source_video_paths:
                raise ValueError("source_video_paths cannot be empty.")
            if source_video_path is None:
                source_video_path = source_video_paths[0]

        # Initialise database
        db.init_db(self.config.db_path)
        db.create_job(
            db_path=self.config.db_path,
            job_id=job_id,
            profile=profile,
            source_path=str(source_video_path),
        )

        job_dirs = file_utils.create_job_dirs(jobs_dir, job_id)
        # Ensure multi-source sources folder exists
        sources_dir = job_dirs.get("sources", job_dirs["root"] / "sources")
        sources_dir.mkdir(parents=True, exist_ok=True)
        job_dirs["sources"] = sources_dir

        job_log_path = jobs_dir / job_id / "pipeline.log"

        def _log_event(stage_name: str, status: str, message: str) -> None:
            now_str = time.strftime("%Y-%m-%d %H:%M:%S")
            log_line = f"[{now_str}] [{stage_name.upper()}] [{status.upper()}] {message}\n"
            try:
                with open(job_log_path, "a", encoding="utf-8") as f:
                    f.write(log_line)
            except Exception:
                pass
            if progress_callback:
                progress_callback(stage_name, status, message)

        context: Dict[str, Any] = {
            "job_id": job_id,
            "source_video_path": source_video_path,
            "source_video_paths": source_video_paths,
            "user_instruction": user_instruction,
            "profile": profile,
            "target_duration": target_duration,
            "aspect_ratio": aspect_ratio,
            "caption_style": caption_style,
            "enable_music": enable_music,
            "job_dirs": job_dirs,
            "jobs_dir": jobs_dir,
            "db_path": self.config.db_path,
        }

        # Stage definitions
        def _stage_ingest(ctx: Dict[str, Any]) -> Dict[str, Any]:
            raw_paths = ctx.get("source_video_paths") or [ctx["source_video_path"]]
            src_paths = [file_utils.validate_video_path(str(p)) for p in raw_paths]
            sources_dir = ctx["job_dirs"]["sources"]
            source_dir = ctx["job_dirs"]["source"]

            copied_sources: List[Path] = []
            source_map: Dict[str, Path] = {}
            metadata_map: Dict[str, VideoMetadata] = {}

            for idx, p in enumerate(src_paths):
                source_id = f"source_{idx}"
                dest = sources_dir / f"{source_id}_{p.name}"
                shutil.copy2(p, dest)
                logger.info("Copied source video to %s", dest)
                copied_sources.append(dest)
                source_map[source_id] = dest

                probe_data = ffmpeg_utils.probe_video(dest)
                meta = VideoMetadata.from_ffprobe(probe_data, str(dest))
                metadata_map[source_id] = meta

                if idx == 0 and sources_dir != source_dir:
                    file_utils.copy_source_video(p, source_dir)

            primary_source = copied_sources[0]
            primary_meta = metadata_map["source_0"]

            # Pre-conform normalization using MediaNormalizer
            from app.editing.normalizer import MediaNormalizer
            _log_event("ingest", "running", f"Conforming {len(source_map)} source video(s) to 1080x1920 30fps...")
            normalizer = MediaNormalizer(target_w=1080, target_h=1920, fps=30)
            conformed_sources: Dict[str, Path] = {}
            for s_id, s_path in source_map.items():
                target_conformed = sources_dir / f"conformed_{s_id}.mp4"
                conformed_path = normalizer.conform_clip(
                    input_path=s_path,
                    output_path=target_conformed,
                    target_w=1080,
                    target_h=1920,
                    fps=30,
                )
                conformed_sources[s_id] = conformed_path

            conformed_primary_probe = ffmpeg_utils.probe_video(conformed_sources["source_0"])
            conformed_primary_meta = VideoMetadata.from_ffprobe(
                conformed_primary_probe, str(conformed_sources["source_0"])
            )

            db.update_job_status(
                ctx["db_path"],
                ctx["job_id"],
                status="RUNNING",
                current_stage="ingest",
                metadata_json=conformed_primary_meta.model_dump_json(),
            )
            return {
                "source_copy": conformed_sources["source_0"],
                "metadata": conformed_primary_meta,
                "source_map": source_map,
                "conformed_sources": conformed_sources,
                "metadata_map": metadata_map,
                "copied_sources": copied_sources,
            }

        def _stage_transcribe(ctx: Dict[str, Any]) -> Dict[str, Any]:
            transcript_json_path = ctx["job_dirs"]["transcripts"] / "transcript.json"
            srt_path = ctx["job_dirs"]["transcripts"] / "transcript.srt"
            ass_path = ctx["job_dirs"]["transcripts"] / "transcript.ass"
            audio_out = ctx["job_dirs"]["intermediate"] / "audio.wav"

            # Check if transcript already exists from a previous stage run
            if transcript_json_path.exists() and transcript_json_path.stat().st_size > 0:
                try:
                    with open(transcript_json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    cached_transcript = Transcript(**data)
                    _log_event(
                        "transcribe",
                        "running",
                        f"Found existing transcript with {len(cached_transcript.segments)} segments, reusing cache...",
                    )
                    return {
                        "transcript": cached_transcript,
                        "audio_path": audio_out,
                        "transcript_json": transcript_json_path,
                        "srt_path": srt_path,
                        "ass_path": ass_path,
                    }
                except Exception as exc:
                    logger.warning("Could not read cached transcript, re-transcribing: %s", exc)

            _log_event("transcribe", "running", "Extracting 16kHz mono audio track from source...")
            ffmpeg_utils.extract_audio(ctx["source_copy"], audio_out)

            _log_event(
                "transcribe",
                "running",
                f"Running MLX Whisper speech recognition ({self.config.whisper_model})...",
            )
            engine = TranscriptionEngine(
                model_name=self.config.whisper_model,
                use_mlx=True,
            )
            transcript: Transcript = engine.transcribe(audio_out)

            transcript_json_path.write_text(transcript.model_dump_json(indent=2), encoding="utf-8")
            generate_srt(transcript, srt_path)
            generate_ass(transcript, ctx["caption_style"], ass_path)

            _log_event(
                "transcribe",
                "running",
                f"Transcription complete: {len(transcript.segments)} speech segments recognized.",
            )

            return {
                "transcript": transcript,
                "audio_path": audio_out,
                "transcript_json": transcript_json_path,
                "srt_path": srt_path,
                "ass_path": ass_path,
            }

        def _stage_analyze(ctx: Dict[str, Any]) -> Dict[str, Any]:
            frames_dir = ctx["job_dirs"]["analysis"] / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            contact_sheet_path = ctx["job_dirs"]["analysis"] / "contact_sheet.jpg"

            existing_frames = list(frames_dir.glob("*.jpg"))
            if existing_frames:
                _log_event(
                    "analyze",
                    "running",
                    f"Found {len(existing_frames)} existing keyframes on disk, reusing visual analysis...",
                )
                return {
                    "keyframes": sorted(existing_frames),
                    "contact_sheet": contact_sheet_path if contact_sheet_path.exists() else None,
                    "scene_changes": [],
                }

            _log_event("analyze", "running", "Extracting periodic visual keyframes and scene changes...")
            frames = extract_keyframes(ctx["source_copy"], frames_dir, interval_seconds=5.0)

            if frames:
                generate_contact_sheet(frames, contact_sheet_path, cols=4)

            scene_changes = self.analyzer.detect_scene_changes(ctx["source_copy"])

            return {
                "keyframes": frames,
                "contact_sheet": contact_sheet_path if contact_sheet_path.exists() else None,
                "scene_changes": scene_changes,
            }

        def _stage_plan(ctx: Dict[str, Any]) -> Dict[str, Any]:
            plan_path = ctx["job_dirs"]["plans"] / "editing_plan.json"

            # Check if valid plan already exists on disk (e.g. from an earlier successful plan stage)
            if plan_path.exists() and plan_path.stat().st_size > 0:
                try:
                    with open(plan_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    cached_plan = EditingPlan(**data)
                    _log_event(
                        "plan",
                        "running",
                        f"Found existing valid editing plan ({len(cached_plan.clips)} clips), reusing plan...",
                    )
                    return {"plan": cached_plan, "plan_path": plan_path}
                except Exception as exc:
                    logger.warning("Could not read cached plan, re-generating: %s", exc)

            instruction_lower = ctx["user_instruction"].lower()
            wants_relentless_silence_cut = (
                "auto-editor" in instruction_lower
                or "silence" in instruction_lower
                or "relentless" in instruction_lower
                or "0.15" in instruction_lower
            )

            plan: Optional[EditingPlan] = None

            if wants_relentless_silence_cut:
                from app.editing.silence_remover import get_speech_segments
                _log_event("plan", "running", "Executing high-retention silence removal (<0.15s pauses) across speech...")
                audio_file = ctx["job_dirs"]["intermediate"] / "audio.wav"
                speech_segs = get_speech_segments(
                    audio_path=audio_file,
                    duration=ctx["metadata"].duration_seconds,
                    min_silence_duration=0.15,
                    silence_threshold_db=-30.0,
                )

                # Safety fallback for 100% silent video
                if not speech_segs:
                    logger.warning("No speech segments detected; preserving full video duration without trimming")
                    speech_segs = [(0.0, ctx["metadata"].duration_seconds)]

                clips: List[Clip] = []
                order = 1
                for s_start, s_end in speech_segs:
                    dur = s_end - s_start
                    if dur < 0.25:
                        continue

                    is_hook = (order == 1)
                    zoom_factor = 1.15 if (is_hook or order % 2 == 1) else 1.0
                    zoom_enabled = (zoom_factor > 1.0)

                    clips.append(Clip(
                        clip_id=f"c{order:03d}",
                        source_start=round(s_start, 2),
                        source_end=round(s_end, 2),
                        order=order,
                        reason="Hook statement" if is_hook else f"Narrative beat {order}",
                        zoom=ZoomEffect(enabled=zoom_enabled, factor=zoom_factor),
                        transition_in="cut",
                        transition_out="cut",
                    ))
                    order += 1


                if clips:
                    plan = EditingPlan(
                        plan_id=str(uuid.uuid4()),
                        job_id=ctx["job_id"],
                        profile=ctx["profile"],
                        aspect_ratio=ctx["aspect_ratio"],
                        target_duration_seconds=ctx["target_duration"],
                        clips=clips,
                        caption_settings=CaptionSettings(
                            enabled=True,
                            style=ctx["caption_style"],
                            font_size=72,
                            word_highlight_color="#FFCC00",
                        ),
                        audio_settings=AudioSettings(
                            normalize=True,
                            fade_in=0.1,
                            fade_out=0.2,
                            music_volume=0.15,
                            duck_music=True,
                        ),
                        hook_summary=ctx["transcript"].segments[0].text if ctx["transcript"].segments else "Opening Hook",
                        edit_rationale="Relentless high-retention cut: trimmed all pauses > 0.15s with alternating 115%/108% punch-in framing.",
                    )
                    _log_event("plan", "running", f"Constructed relentless cut plan ({len(clips)} clips, {plan.total_duration:.1f}s)")

            if plan is None:
                def _editorial_logger(sub_msg: str) -> None:
                    _log_event("plan", "running", sub_msg)

                plan = self.agent.analyze_and_plan(
                    transcript=ctx["transcript"],
                    video_metadata=ctx["metadata"],
                    user_instruction=ctx["user_instruction"],
                    profile=ctx["profile"],
                    target_duration=ctx["target_duration"],
                    aspect_ratio=ctx["aspect_ratio"],
                    job_id=ctx["job_id"],
                    log_callback=_editorial_logger,
                )

            # Update caption style preference in plan
            plan.caption_settings.style = ctx["caption_style"]

            # Handle background music preference
            if ctx.get("enable_music"):
                assets_music = Path(__file__).resolve().parent.parent.parent / "assets" / "music"
                music_files = list(assets_music.glob("*.wav")) + list(assets_music.glob("*.mp3"))
                if music_files:
                    plan.audio_settings.background_music_path = str(music_files[0])
                    logger.info("Attached background music track: %s", music_files[0].name)

            # Semantic B-Roll Keyword Matching (Feature 10)
            if not plan.broll_overlays:
                try:
                    from app.analysis.broll_matcher import BrollMatcher
                    from app.analysis.broll_pool import BrollCatalog

                    catalog = BrollCatalog(auto_provision=True)
                    if catalog.total_count > 0:
                        matcher = BrollMatcher(provider=self.provider, catalog=catalog)
                        overlays = matcher.match_broll(plan=plan, transcript=ctx.get("transcript"))
                        plan.broll_overlays = overlays
                        if overlays:
                            tot_ov_dur = sum(o.duration for o in overlays)
                            pct = (tot_ov_dur / plan.total_duration * 100) if plan.total_duration > 0 else 0
                            _log_event(
                                "plan",
                                "running",
                                f"Placed {len(overlays)} B-roll overlay(s) on timeline ({tot_ov_dur:.1f}s total / {pct:.0f}% of edit).",
                            )
                except Exception as exc:
                    logger.warning("B-roll matching skipped due to error: %s", exc)

            plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")

            return {"plan": plan, "plan_path": plan_path}

        def _stage_edit(ctx: Dict[str, Any]) -> Dict[str, Any]:
            sources_to_use = ctx.get("conformed_sources") or ctx.get("source_map") or ctx["source_copy"]
            rough_cut = self.editor.execute_plan(
                plan=ctx["plan"],
                source_path=sources_to_use,
                job_dirs=ctx["job_dirs"],
                source_paths=ctx.get("conformed_sources") or ctx.get("source_map"),
            )
            return {"rough_cut": rough_cut}

        def _stage_render(ctx: Dict[str, Any]) -> Dict[str, Any]:
            render_result = self.renderer.render_final_video(
                rough_cut=ctx["rough_cut"],
                plan=ctx["plan"],
                job_dirs=ctx["job_dirs"],
                transcript=ctx.get("transcript"),
                output_name="final.mp4",
            )
            # render_final_video now returns (path, sfx_events)
            if isinstance(render_result, tuple):
                final_mp4, sfx_events = render_result
            else:
                final_mp4 = render_result
                sfx_events = getattr(self.renderer, "last_sfx_events", [])

            # Export multi-track FCPXML timeline for DaVinci Resolve
            fcpxml_path = ctx["job_dirs"]["exports"] / "timeline.fcpxml"
            try:
                from pathlib import Path as _Path
                music_path = None
                audio_settings = ctx["plan"].audio_settings
                if audio_settings.background_music_path:
                    _mp = _Path(audio_settings.background_music_path)
                    if _mp.exists():
                        music_path = _mp

                ass_path = ctx["job_dirs"].get("transcripts", ctx["job_dirs"]["exports"]) / "captions.ass"

                generate_multitrack_fcpxml(
                    plan=ctx["plan"],
                    source_video_path=ctx["source_copy"],
                    metadata=ctx["metadata"],
                    output_path=fcpxml_path,
                    project_name=f"Marvel Cut {ctx['job_id'][:8]}",
                    music_path=music_path,
                    sfx_events=sfx_events or [],
                    broll_overlays=ctx["plan"].broll_overlays or [],
                    ass_subtitle_path=ass_path if ass_path.exists() else None,
                )
                _log_event("render", "running", f"Exported multi-track DaVinci Resolve timeline: {fcpxml_path.name}")
            except Exception as exc:
                logger.warning("Multi-track FCPXML failed (%s); trying legacy generator", exc)
                try:
                    generate_fcpxml(
                        plan=ctx["plan"],
                        source_video_path=ctx["source_copy"],
                        metadata=ctx["metadata"],
                        output_path=fcpxml_path,
                        project_name=f"Studio Cut {ctx['job_id'][:8]}",
                    )
                    _log_event("render", "running", f"Exported DaVinci Resolve timeline (legacy): {fcpxml_path.name}")
                except Exception as exc2:
                    logger.warning("FCPXML generation skipped: %s", exc2)

            return {"final_video": final_mp4, "fcpxml_path": fcpxml_path}

        def _stage_quality(ctx: Dict[str, Any]) -> Dict[str, Any]:
            report = self.quality_checker.check(ctx["final_video"], ctx["plan"])
            report_path = ctx["job_dirs"]["exports"] / "quality_report.json"
            report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            return {"quality_report": report, "quality_report_path": report_path}

        def _stage_preview(ctx: Dict[str, Any]) -> Dict[str, Any]:
            preview_mp4 = ctx["job_dirs"]["previews"] / "preview.mp4"
            generate_preview(ctx["final_video"], preview_mp4)

            db.update_job_status(
                ctx["db_path"],
                ctx["job_id"],
                status="DONE",
                current_stage="preview",
            )
            return {"preview_video": preview_mp4}

        stages = [
            StageRunner(name="ingest", fn=_stage_ingest, critical=True, description="Ingesting and inspecting video"),
            StageRunner(name="transcribe", fn=_stage_transcribe, critical=True, description="Transcribing speech with Whisper"),
            StageRunner(name="analyze", fn=_stage_analyze, critical=False, description="Analyzing frames and scene changes"),
            StageRunner(name="plan", fn=_stage_plan, critical=True, description="AI editorial analysis & cut planning"),
            StageRunner(name="edit", fn=_stage_edit, critical=True, description="Cutting, joining, and processing audio"),
            StageRunner(name="render", fn=_stage_render, critical=True, description="Rendering final video and burning captions"),
            StageRunner(name="quality", fn=_stage_quality, critical=False, description="Running automated quality checks"),
            StageRunner(name="preview", fn=_stage_preview, critical=True, description="Generating web-optimized preview"),
        ]

        try:
            return run_stages(stages, context, _log_event)
        except Exception as exc:
            db.update_job_status(
                self.config.db_path,
                job_id,
                status="FAILED",
                error=str(exc),
            )
            raise
