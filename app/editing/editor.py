"""
app/editing/editor.py — FFmpeg-based video editor that executes EditingPlans.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

from app.schemas.editing_plan import AspectRatio, Clip, EditingPlan
from app.utils import ffmpeg_utils
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Aspect ratio → (width, height)
_ASPECT_RATIO_DIMS = {
    AspectRatio.VERTICAL_9_16: (1080, 1920),
    AspectRatio.LANDSCAPE_16_9: (1920, 1080),
    AspectRatio.SQUARE_1_1: (1080, 1080),
}


def _resolve_clip_source(
    clip: Clip,
    source_input: Union[Path, str, Dict[str, Union[Path, str]], List[Union[Path, str]]],
    source_paths: Optional[Dict[str, Union[Path, str]]] = None,
) -> Path:
    """
    Resolve the source media file path for a given clip.

    Checks:
    1. Explicit source_paths dict matching clip.source_id.
    2. clip.source_path attribute if present and existing on disk.
    3. source_input if dict matching clip.source_id or numeric index.
    4. source_input if list indexing by numeric suffix of clip.source_id.
    5. source_input if single Path / string.
    """
    source_id = getattr(clip, "source_id", "source_0") or "source_0"

    # 1. Check explicit source_paths dict
    if source_paths and source_id in source_paths:
        return Path(source_paths[source_id])

    # 2. Check clip's direct source_path attribute
    clip_sp = getattr(clip, "source_path", None)
    if clip_sp:
        p = Path(clip_sp)
        if p.exists():
            return p

    # 3. source_input is a Dict
    if isinstance(source_input, dict):
        if source_id in source_input:
            return Path(source_input[source_id])
        # Check source_0 vs 0
        if source_id.startswith("source_"):
            num_key = source_id.replace("source_", "")
            if num_key in source_input:
                return Path(source_input[num_key])
        elif f"source_{source_id}" in source_input:
            return Path(source_input[f"source_{source_id}"])
        if "default" in source_input:
            return Path(source_input["default"])
        if len(source_input) == 1:
            return Path(next(iter(source_input.values())))
        raise KeyError(
            f"Clip '{clip.clip_id}' references source_id '{source_id}', "
            f"not found in available sources: {list(source_input.keys())}"
        )

    # 4. source_input is a List
    if isinstance(source_input, list):
        if not source_input:
            raise ValueError("source_input list is empty")
        idx = 0
        if source_id.startswith("source_"):
            try:
                idx = int(source_id.split("_")[1])
            except (ValueError, IndexError):
                idx = 0
        elif source_id.isdigit():
            idx = int(source_id)
        if 0 <= idx < len(source_input):
            return Path(source_input[idx])
        return Path(source_input[0])

    # 5. source_input is a single Path or str
    if isinstance(source_input, (str, Path)):
        return Path(source_input)

    raise TypeError(f"Unsupported source_input type: {type(source_input)}")


class VideoEditor:
    """
    Executes an :class:`~app.schemas.editing_plan.EditingPlan` by cutting clips,
    applying effects, concatenating them, and processing audio.
    """

    def execute_plan(
        self,
        plan: EditingPlan,
        source_path: Union[Path, str, Dict[str, Union[Path, str]], List[Union[Path, str]]],
        job_dirs: Dict[str, Path],
        source_paths: Optional[Dict[str, Union[Path, str]]] = None,
    ) -> Path:
        """
        Execute the editing plan and return the path to the rough cut video.

        Supports multi-source cutting where clips reference different source_id entries.

        Steps:
        1. Cut individual clips from their resolved source videos.
        2. Apply zoom effect if requested.
        3. Concatenate all clips.
        4. Normalise audio.
        5. Apply audio fade in/out.

        Parameters
        ----------
        plan:
            Validated editing plan from the editorial agent.
        source_path:
            Path, dict of paths, or list of paths to source video(s).
        job_dirs:
            Dict mapping directory names to paths (from :func:`create_job_dirs`).
        source_paths:
            Optional mapping from source_id to resolved video file path.

        Returns
        -------
        Path
            Path to the rough cut video in ``job_dirs["intermediate"]``.
        """
        intermediate_dir = job_dirs["intermediate"]
        sorted_clips = sorted(plan.clips, key=lambda c: c.order)

        # Retrieve source_map from plan if available
        combined_source_paths = dict(source_paths or {})
        plan_source_map = getattr(plan, "source_map", None)
        if plan_source_map and isinstance(plan_source_map, dict):
            for k, v in plan_source_map.items():
                combined_source_paths.setdefault(k, v)

        # Check if any clip has J-cut or L-cut offsets
        has_split_edits = any(
            (getattr(c, "audio_lead", 0.0) > 0.0 or getattr(c, "audio_tail", 0.0) > 0.0)
            for c in sorted_clips
        )

        if has_split_edits:
            rough_cut = self._execute_split_edit_plan(
                sorted_clips=sorted_clips,
                source_path=source_path,
                combined_source_paths=combined_source_paths,
                intermediate_dir=intermediate_dir,
            )
        else:
            cut_paths: List[Path] = []
            for clip in sorted_clips:
                # 1. Resolve source video for this specific clip
                clip_source = _resolve_clip_source(clip, source_path, combined_source_paths)
                if not clip_source.exists():
                    raise FileNotFoundError(
                        f"Resolved source file for clip {clip.clip_id} does not exist: {clip_source}"
                    )

                # 2. Cut segment (with 10ms micro-fade by default)
                clip_path = intermediate_dir / f"clip_{clip.clip_id}.mp4"
                ffmpeg_utils.cut_segment(
                    input_path=clip_source,
                    start=clip.source_start,
                    end=clip.source_end,
                    output_path=clip_path,
                )
                logger.info(
                    "Cut clip %s [%.2f–%.2f] from %s → %s",
                    clip.clip_id,
                    clip.source_start,
                    clip.source_end,
                    clip_source.name,
                    clip_path.name,
                )

                # 3. Zoom effect
                if clip.zoom.enabled:
                    try:
                        zoomed_path = intermediate_dir / f"clip_{clip.clip_id}_zoomed.mp4"
                        ffmpeg_utils.apply_zoom_effect(
                            input_path=clip_path,
                            output_path=zoomed_path,
                            zoom_factor=clip.zoom.factor,
                            duration=clip.duration,
                        )
                        if zoomed_path.exists() and zoomed_path.stat().st_size > 0:
                            clip_path = zoomed_path
                    except Exception as exc:
                        logger.warning("Zoom effect skipped on clip %s: %s", clip.clip_id, exc)

                cut_paths.append(clip_path)

            # 4. Concatenate (robust, guaranteed monotonic DTS)
            if len(cut_paths) == 1:
                rough_cut = cut_paths[0]
            else:
                concat_path = intermediate_dir / "rough_cut_raw.mp4"
                ffmpeg_utils.concat_clips(cut_paths, concat_path)
                rough_cut = concat_path

        # 5. Audio normalisation
        audio = plan.audio_settings
        if audio.normalize:
            norm_path = intermediate_dir / "rough_cut_norm.mp4"
            ffmpeg_utils.normalize_audio(rough_cut, norm_path)
            rough_cut = norm_path

        # 6. Audio fades
        faded_path = intermediate_dir / "rough_cut_faded.mp4"
        ffmpeg_utils.apply_audio_fade(
            rough_cut,
            faded_path,
            fade_in=audio.fade_in,
            fade_out=audio.fade_out,
        )
        rough_cut = faded_path

        logger.info("Rough cut complete → %s", rough_cut)
        return rough_cut

    def _execute_split_edit_plan(
        self,
        sorted_clips: List[Clip],
        source_path: Union[Path, str, Dict[str, Union[Path, str]], List[Union[Path, str]]],
        combined_source_paths: Dict[str, Union[Path, str]],
        intermediate_dir: Path,
    ) -> Path:
        """
        Execute an EditingPlan containing J-cuts (audio_lead) or L-cuts (audio_tail)
        with sample-accurate audio track delay alignment and multi-track mixing.
        """
        n = len(sorted_clips)
        v_cuts: List[Path] = []
        v_offsets: List[float] = []
        cum_t = 0.0

        for clip in sorted_clips:
            v_offsets.append(cum_t)
            cum_t += clip.duration

            clip_source = _resolve_clip_source(clip, source_path, combined_source_paths)
            if not clip_source.exists():
                raise FileNotFoundError(
                    f"Resolved source file for clip {clip.clip_id} does not exist: {clip_source}"
                )

            clip_v_path = intermediate_dir / f"clip_{clip.clip_id}_v.mp4"
            ffmpeg_utils.cut_segment(
                input_path=clip_source,
                start=clip.source_start,
                end=clip.source_end,
                output_path=clip_v_path,
                micro_fade=False,
            )

            if clip.zoom.enabled:
                try:
                    zoomed_path = intermediate_dir / f"clip_{clip.clip_id}_v_zoomed.mp4"
                    ffmpeg_utils.apply_zoom_effect(
                        input_path=clip_v_path,
                        output_path=zoomed_path,
                        zoom_factor=clip.zoom.factor,
                        duration=clip.duration,
                    )
                    if zoomed_path.exists() and zoomed_path.stat().st_size > 0:
                        clip_v_path = zoomed_path
                except Exception as exc:
                    logger.warning("Zoom effect skipped on clip %s: %s", clip.clip_id, exc)

            v_cuts.append(clip_v_path)

        # Concatenate master video track
        if len(v_cuts) == 1:
            video_concat_path = v_cuts[0]
        else:
            video_concat_path = intermediate_dir / "rough_video_concat.mp4"
            ffmpeg_utils.concat_clips(v_cuts, video_concat_path)

        # Process audio tracks with clamped offsets and fades
        ffmpeg_bin = ffmpeg_utils.find_ffmpeg()
        a_cuts: List[Path] = []
        delays_ms: List[int] = []

        for i, clip in enumerate(sorted_clips):
            clip_source = _resolve_clip_source(clip, source_path, combined_source_paths)
            v_start = v_offsets[i]

            # 1. Clamp audio_lead
            if i == 0:
                eff_lead = 0.0
            else:
                max_lead = sorted_clips[i - 1].duration
                max_lead_src = max(0.0, clip.source_start)
                raw_lead = max(0.0, getattr(clip, "audio_lead", 0.0) or 0.0)
                eff_lead = min(raw_lead, max_lead, max_lead_src)

            # 2. Clamp audio_tail
            if i == n - 1:
                eff_tail = 0.0
            else:
                max_tail = sorted_clips[i + 1].duration
                try:
                    p = ffmpeg_utils.probe_video(clip_source)
                    src_dur = float(p.get("format", {}).get("duration", clip.source_end + 10.0))
                except Exception:
                    src_dur = clip.source_end + 10.0
                max_tail_src = max(0.0, src_dur - clip.source_end)
                raw_tail = max(0.0, getattr(clip, "audio_tail", 0.0) or 0.0)
                eff_tail = min(raw_tail, max_tail, max_tail_src)

            src_a_start = max(0.0, clip.source_start - eff_lead)
            src_a_end = clip.source_end + eff_tail
            a_dur = max(0.001, src_a_end - src_a_start)

            t_a_start = max(0.0, v_start - eff_lead)
            delay_ms = max(0, int(round(t_a_start * 1000)))
            delays_ms.append(delay_ms)

            # Determine boundary fade curves
            fade_in_d = eff_lead if eff_lead > 0.0 else min(0.01, a_dur / 2.0)
            fade_out_d = eff_tail if eff_tail > 0.0 else min(0.01, a_dur / 2.0)
            fade_out_st = max(0.0, a_dur - fade_out_d)

            af = (
                f"afade=t=in:st=0:d={fade_in_d:.4f},"
                f"afade=t=out:st={fade_out_st:.4f}:d={fade_out_d:.4f}"
            )

            a_out = intermediate_dir / f"clip_{clip.clip_id}_split_audio.wav"
            try:
                probe = ffmpeg_utils.probe_video(clip_source)
                has_audio = any(s.get("codec_type") == "audio" for s in probe.get("streams", []))
            except Exception:
                has_audio = True

            if has_audio:
                args = [
                    ffmpeg_bin, "-y",
                    "-ss", str(src_a_start),
                    "-to", str(src_a_end),
                    "-i", str(clip_source),
                    "-af", af,
                    "-vn",
                    "-c:a", "pcm_s16le",
                    "-ar", "48000",
                    "-ac", "2",
                    str(a_out),
                ]
            else:
                args = [
                    ffmpeg_bin, "-y",
                    "-f", "lavfi",
                    "-i", "anullsrc=r=48000:cl=stereo",
                    "-t", f"{a_dur:.4f}",
                    "-c:a", "pcm_s16le",
                    str(a_out),
                ]
            ffmpeg_utils._run(args)
            a_cuts.append(a_out)

        # Mix all audio clips using adelay and amix(normalize=0)
        mixed_audio = intermediate_dir / "rough_audio_mixed.m4a"
        if len(a_cuts) == 1:
            mixed_audio = a_cuts[0]
        else:
            inputs = []
            for p in a_cuts:
                inputs.extend(["-i", str(p)])

            filter_parts = []
            for idx, d_ms in enumerate(delays_ms):
                filter_parts.append(f"[{idx}:a]adelay={d_ms}|{d_ms}[d{idx}]")
            amix_ins = "".join(f"[d{idx}]" for idx in range(len(delays_ms)))
            filter_parts.append(f"{amix_ins}amix=inputs={len(delays_ms)}:duration=longest:normalize=0[aout]")
            filter_complex = ";".join(filter_parts)

            args = [
                ffmpeg_bin, "-y",
                *inputs,
                "-filter_complex", filter_complex,
                "-map", "[aout]",
                "-c:a", "aac",
                "-b:a", "192k",
                "-ar", "48000",
                "-ac", "2",
                str(mixed_audio),
            ]
            ffmpeg_utils._run(args)

        # Mux master video and mixed audio with -shortest to clamp container length
        rough_cut = intermediate_dir / "rough_cut_raw.mp4"
        args = [
            ffmpeg_bin, "-y",
            "-i", str(video_concat_path),
            "-i", str(mixed_audio),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-ar", "48000",
            "-ac", "2",
            "-b:a", "192k",
            "-shortest",
            str(rough_cut),
        ]
        ffmpeg_utils._run(args)
        return rough_cut


    def apply_aspect_ratio(
        self,
        input_path: Path,
        aspect_ratio: AspectRatio,
        output_path: Path,
    ) -> Path:
        """
        Crop/pad a video to the target aspect ratio using FFmpeg.

        Parameters
        ----------
        input_path:
            Source video file.
        aspect_ratio:
            Target aspect ratio enum value.
        output_path:
            Destination file path.

        Returns
        -------
        Path
            Path to the aspect-ratio-converted video.
        """
        width, height = _ASPECT_RATIO_DIMS.get(aspect_ratio, (1920, 1080))
        return ffmpeg_utils.render_final(
            input_path=input_path,
            output_path=output_path,
            width=width,
            height=height,
        )
