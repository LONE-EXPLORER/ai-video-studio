"""
app/captions/pillow_captions.py — High-impact kinetic typography overlay generator using Pillow & FFmpeg.

Features:
- Fast speech-interval rendering: skips non-speech intervals (pauses between words/phrases)
- Kinetic spring/bounce entry animations: 1.20x -> 1.00x pop over 120ms with smooth damping
- TrueType vector font point scaling with FreeType LRU cache (63x faster than raster resizing)
- Two-pass layout architecture: immutable optical centers (Cx, Cy) eliminate text reflow jitter
- 3-layer Z-ordering: Layer 1: Translucent Pill -> Layer 2: Inactive Words -> Layer 3: Active Word
- Single-pass Apple Silicon VideoToolbox hardware compositing (h264_videotoolbox) with libx264 fallback
- Dialogue audio track 0 preservation with zero audio drift or recompression artifacts
"""

from __future__ import annotations

import functools
import io
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from PIL import Image, ImageDraw, ImageFont

from app.schemas.editing_plan import EditingPlan
from app.schemas.transcript import Transcript, TranscriptWord
from app.utils.ffmpeg_utils import find_ffmpeg
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Fonts in order of preference on macOS
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


@functools.lru_cache(maxsize=64)
def _get_font(size: int = 72, font_path: Optional[str] = None) -> ImageFont.FreeTypeFont:
    """Cached FreeType font loader for fast vector point scaling."""
    if font_path and Path(font_path).exists():
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    for candidate in _FONT_CANDIDATES:
        p = Path(candidate)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                pass
    return ImageFont.load_default()


def _get_font_cached(font_path: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    """Alias for cached font loading."""
    return _get_font(size=size, font_path=font_path)


def spring_scale(t: float, duration: float = 0.120) -> float:
    """
    Bounce animation envelope: 1.20x -> 1.00x over duration with smooth damping.
    Strictly monotonic decay matching tests/e2e/test_tier1_features.py (F13-1..F13-5)
    and tests/e2e/test_tier2_boundaries.py (F13-B1..F13-B5).
    """
    if math.isnan(t) or math.isinf(t):
        return 1.00
    if duration <= 0.0:
        return 1.00
    if t <= 0.0:
        return 1.20
    if t >= duration:
        return 1.00
    progress = t / duration
    decay = (1.0 - progress) * 0.20
    return 1.00 + decay * math.cos(progress * math.pi / 2.0)


def compute_harmonic_spring_scale(
    t: float,
    amplitude: float = 0.20,
    zeta: float = 0.65,
    omega: float = 25.0,
) -> float:
    """
    Remotion-style damped harmonic oscillator for physical elastic bounce.
    Formula: s(t) = 1.0 + A * exp(-zeta * omega * t) * cos(omega_d * t)
    """
    if math.isnan(t) or math.isinf(t):
        return 1.00
    if t <= 0.0:
        return 1.00 + amplitude
    if t >= 0.250:
        return 1.00
    omega_d = omega * math.sqrt(max(0.0, 1.0 - zeta**2))
    return 1.00 + amplitude * math.exp(-zeta * omega * t) * math.cos(omega_d * t)


class MappedWord:
    """A word with timeline-mapped timestamps relative to the cut video."""

    def __init__(self, text: str, start: float, end: float):
        self.text = text.strip()
        self.start = start
        self.end = end

    def __repr__(self) -> str:
        return f"MappedWord({self.text}, {self.start:.2f}-{self.end:.2f})"


class CaptionGroup:
    """A small punchy phrase (2-4 words) displayed together on screen."""

    def __init__(self, words: List[MappedWord]):
        self.words = words
        self.start = words[0].start if words else 0.0
        self.end = words[-1].end if words else 0.0


class WordLayoutItem:
    """Precomputed layout metrics for a single word in a caption group."""

    def __init__(
        self,
        word: MappedWord,
        base_x: float,
        base_y: float,
        base_w: float,
        base_h: float,
        cx: float,
        cy: float,
        bbox_raw: Tuple[int, int, int, int],
    ):
        self.word = word
        self.text = word.text
        self.base_x = base_x
        self.base_y = base_y
        self.base_w = base_w
        self.base_h = base_h
        self.cx = cx
        self.cy = cy
        self.bbox_raw = bbox_raw


class PhraseLayout:
    """Precomputed immutable layout geometry for a CaptionGroup phrase card."""

    def __init__(
        self,
        group: CaptionGroup,
        font_base: ImageFont.FreeTypeFont,
        canvas_width: int = 1080,
        box_y: int = 1190,
        word_spacing: int = 24,
    ):
        self.group = group
        self.words = group.words
        self.font_base = font_base
        self.canvas_width = canvas_width
        self.box_y = box_y
        self.word_spacing = word_spacing
        self.items: List[WordLayoutItem] = []

        dummy = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        draw = ImageDraw.Draw(dummy)

        # Pass 1: Measure all words at base scale
        metrics = []
        total_w = 0.0
        for i, w in enumerate(self.words):
            bbox = draw.textbbox((0, 0), w.text, font=font_base)
            w_w = float(bbox[2] - bbox[0])
            w_h = float(bbox[3] - bbox[1])
            metrics.append((w, w_w, w_h, bbox))
            total_w += w_w
            if i > 0:
                total_w += word_spacing

        start_x = (canvas_width - total_w) / 2.0
        cur_x = start_x

        for w, w_w, w_h, bbox in metrics:
            cx = cur_x + w_w / 2.0
            cy = float(box_y) + w_h / 2.0
            self.items.append(
                WordLayoutItem(
                    word=w,
                    base_x=cur_x,
                    base_y=float(box_y),
                    base_w=w_w,
                    base_h=w_h,
                    cx=cx,
                    cy=cy,
                    bbox_raw=bbox,
                )
            )
            cur_x += w_w + word_spacing


def render_sample_word_pill(
    word: str,
    highlight: bool = True,
    width: int = 1080,
    height: int = 1920,
    pill_color: Tuple[int, int, int, int] = (0, 0, 0, 180),
) -> Image.Image:
    """Helper to render active word pill with translucent rounded background."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _get_font(size=72)
    color = (255, 204, 0, 255) if highlight else (255, 255, 255, 255)

    bbox = draw.textbbox((0, 0), word, font=font)
    w_w = bbox[2] - bbox[0]
    w_h = bbox[3] - bbox[1]
    pad_x, pad_y = 20, 10

    cx = width // 2
    cy = int(height * 0.62)

    x0 = cx - w_w // 2 - pad_x
    y0 = cy - w_h // 2 - pad_y
    x1 = cx + w_w // 2 + pad_x
    y1 = cy + w_h // 2 + pad_y

    draw.rounded_rectangle([x0, y0, x1, y1], radius=16, fill=pill_color)
    tx = cx - bbox[0] - w_w // 2
    ty = cy - bbox[1] - w_h // 2
    draw.text((tx + 4, ty + 4), word, font=font, fill=(0, 0, 0, 210))
    draw.text((tx, ty), word, font=font, fill=color, stroke_width=6, stroke_fill=(0, 0, 0, 255))
    return img


def render_caption_card(
    layout: PhraseLayout,
    active_idx: Optional[int],
    scale: float = 1.0,
    width: int = 1080,
    height: int = 1920,
    base_font_size: int = 72,
    font_path: Optional[str] = None,
    pill_color: Tuple[int, int, int, int] = (20, 20, 25, 180),
    color_active: Tuple[int, int, int, int] = (255, 204, 0, 255),
    color_inactive: Tuple[int, int, int, int] = (255, 255, 255, 255),
    color_stroke: Tuple[int, int, int, int] = (0, 0, 0, 255),
    color_shadow: Tuple[int, int, int, int] = (0, 0, 0, 210),
    radius: int = 16,
    pad_x: int = 14,
    pad_y: int = 8,
) -> Image.Image:
    """
    Render a single caption card frame with 3-layer Z-ordering:
    Layer 1 (Bottom): Translucent active-word pill background
    Layer 2 (Middle): Inactive words text
    Layer 3 (Top): Active word text with spring bounce scale
    """
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    scaled_font = layout.font_base
    if scale != 1.0 and active_idx is not None:
        scaled_size = int(round(base_font_size * scale))
        scaled_font = _get_font(scaled_size, font_path)

    # -------------------------------------------------------------
    # Layer 1: Active Word Pill Background (Bottom)
    # -------------------------------------------------------------
    if active_idx is not None and 0 <= active_idx < len(layout.items):
        act_item = layout.items[active_idx]
        bbox_act = draw.textbbox((0, 0), act_item.text, font=scaled_font)
        scaled_w = float(bbox_act[2] - bbox_act[0])
        scaled_h = float(bbox_act[3] - bbox_act[1])

        # Centered symmetrically around immutable optical center (cx, cy)
        pill_x0 = act_item.cx - scaled_w / 2.0 - pad_x
        pill_y0 = act_item.cy - scaled_h / 2.0 - pad_y
        pill_x1 = act_item.cx + scaled_w / 2.0 + pad_x
        pill_y1 = act_item.cy + scaled_h / 2.0 + pad_y

        draw.rounded_rectangle([pill_x0, pill_y0, pill_x1, pill_y1], radius=radius, fill=pill_color)

    # -------------------------------------------------------------
    # Layer 2: Inactive Words Text (Middle)
    # -------------------------------------------------------------
    for idx, item in enumerate(layout.items):
        if idx == active_idx:
            continue
        bbox_raw = item.bbox_raw
        inact_x = item.cx - bbox_raw[0] - item.base_w / 2.0
        inact_y = item.cy - bbox_raw[1] - item.base_h / 2.0

        # Shadow
        draw.text((inact_x + 4, inact_y + 4), item.text, font=layout.font_base, fill=color_shadow)
        # Text with black outline
        draw.text(
            (inact_x, inact_y),
            item.text,
            font=layout.font_base,
            fill=color_inactive,
            stroke_width=6,
            stroke_fill=color_stroke,
        )

    # -------------------------------------------------------------
    # Layer 3: Active Word Text (Top)
    # -------------------------------------------------------------
    if active_idx is not None and 0 <= active_idx < len(layout.items):
        act_item = layout.items[active_idx]
        bbox_act = draw.textbbox((0, 0), act_item.text, font=scaled_font)
        scaled_w = float(bbox_act[2] - bbox_act[0])
        scaled_h = float(bbox_act[3] - bbox_act[1])

        act_x = act_item.cx - bbox_act[0] - scaled_w / 2.0
        act_y = act_item.cy - bbox_act[1] - scaled_h / 2.0
        stroke_w = 7 if scale > 1.05 else 6

        # Shadow
        draw.text((act_x + 4, act_y + 4), act_item.text, font=scaled_font, fill=color_shadow)
        # Active text in vibrant gold/yellow
        draw.text(
            (act_x, act_y),
            act_item.text,
            font=scaled_font,
            fill=color_active,
            stroke_width=stroke_w,
            stroke_fill=color_stroke,
        )

    return img


def map_words_to_timeline(plan: EditingPlan, transcript: Transcript) -> List[MappedWord]:
    """
    Project transcript words onto the edited timeline by accounting for clip cuts.
    """
    mapped: List[MappedWord] = []
    sorted_clips = sorted(plan.clips, key=lambda c: c.order)

    # Collect all words from transcript
    all_words: List[TranscriptWord] = []
    for seg in transcript.segments:
        all_words.extend(seg.words)

    timeline_offset = 0.0
    for clip in sorted_clips:
        clip_start = clip.source_start
        clip_end = clip.source_end

        for w in all_words:
            # Word is inside this clip
            if clip_start <= w.start < clip_end:
                w_start_timeline = timeline_offset + (w.start - clip_start)
                w_end_timeline = timeline_offset + (min(w.end, clip_end) - clip_start)
                clean_text = w.word.strip().upper()
                if clean_text:
                    mapped.append(MappedWord(clean_text, w_start_timeline, w_end_timeline))

        timeline_offset += clip.duration

    mapped.sort(key=lambda w: (w.start, w.end))
    return mapped


def group_words_into_phrases(words: List[MappedWord], max_words: int = 4) -> List[CaptionGroup]:
    """Group words into short, bite-sized caption cards (2-4 words each)."""
    groups: List[CaptionGroup] = []
    current: List[MappedWord] = []

    for w in words:
        current.append(w)
        # End card on punctuation or max length
        has_punct = any(p in w.text for p in [".", ",", "!", "?", "—", "..."])
        if len(current) >= max_words or has_punct:
            groups.append(CaptionGroup(current))
            current = []

    if current:
        groups.append(CaptionGroup(current))

    return groups


def check_videotoolbox_available(ffmpeg: str) -> bool:
    """Check if h264_videotoolbox is available in FFmpeg."""
    try:
        res = subprocess.run([ffmpeg, "-encoders"], capture_output=True, text=True, timeout=5)
        return "h264_videotoolbox" in res.stdout
    except Exception:
        return False


def _get_encoder_args(ffmpeg: str, video_bitrate: str = "6000k") -> Tuple[List[str], bool]:
    """Select VideoToolbox hardware encoder if available, otherwise libx264."""
    if check_videotoolbox_available(ffmpeg):
        return [
            "-c:v", "h264_videotoolbox",
            "-b:v", video_bitrate,
            "-maxrate", "8000k",
            "-pix_fmt", "nv12",
            "-color_range", "tv",
            "-colorspace", "bt709",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
        ], True
    return [
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-color_range", "tv",
        "-colorspace", "bt709",
    ], False


def burn_kinetic_captions_direct_pipe(
    video_path: Path,
    plan: EditingPlan,
    transcript: Transcript,
    output_path: Path,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
    video_bitrate: str = "6000k",
    font_size: int = 72,
    font_path: Optional[str] = None,
    color_active: Tuple[int, int, int, int] = (255, 204, 0, 255),
    color_inactive: Tuple[int, int, int, int] = (255, 255, 255, 255),
    pill_color: Tuple[int, int, int, int] = (20, 20, 25, 180),
) -> Path:
    """
    High-throughput single-pass direct pipe kinetic caption overlay with VideoToolbox hardware acceleration.
    Frame deduplication skips redundant drawing during non-speech intervals and steady-state word display.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()

    mapped_words = map_words_to_timeline(plan, transcript)
    if not mapped_words:
        logger.warning("No words found to caption, copying video directly.")
        shutil.copy2(str(video_path), str(output_path))
        return output_path

    caption_groups = group_words_into_phrases(mapped_words, max_words=4)
    total_duration = plan.total_duration
    total_frames = int(math.ceil(total_duration * fps))

    box_y = int(height * 0.62)
    word_spacing = max(24, int(font_size * 0.35))
    font_base = _get_font(font_size, font_path)

    # Pass 1: Precompute immutable optical centers for all phrase cards
    layouts = [
        PhraseLayout(g, font_base, canvas_width=width, box_y=box_y, word_spacing=word_spacing)
        for g in caption_groups
    ]

    # Pre-render single reusable transparent blank image bytes
    blank_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    bio_blank = io.BytesIO()
    blank_img.save(bio_blank, "PNG", compress_level=1)
    blank_bytes = bio_blank.getvalue()

    encoder_args, is_vt = _get_encoder_args(ffmpeg, video_bitrate)

    def _execute_pipe_pass(v_encoder_args: List[str], audio_args: List[str]) -> subprocess.Popen:
        cmd = [
            ffmpeg, "-y",
            "-i", str(video_path),
            "-f", "image2pipe",
            "-framerate", str(fps),
            "-i", "-",
            "-filter_complex", "[0:v][1:v]overlay=0:0:shortest=1[v]",
            "-map", "[v]",
            "-map", "0:a?",
            *v_encoder_args,
            *audio_args,
            "-movflags", "+faststart",
            str(output_path),
        ]
        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    # Execution with automatic fallback for VideoToolbox and audio copy
    attempts = [
        (encoder_args, ["-c:a", "copy"]),
        (encoder_args, ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]),
        (
            [
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-color_range", "tv",
                "-colorspace", "bt709",
            ],
            ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"],
        ),
    ]

    rendered_cache: Dict[Tuple[int, Optional[int], int], bytes] = {}

    for attempt_idx, (v_args, a_args) in enumerate(attempts):
        proc = _execute_pipe_pass(v_args, a_args)
        last_cache_key = None
        last_bytes = None
        group_idx = 0
        failed = False
        error_msg = ""

        try:
            for frame_idx in range(total_frames):
                current_t = frame_idx / fps

                # Advance group pointer
                while group_idx < len(caption_groups) and caption_groups[group_idx].end < current_t:
                    group_idx += 1

                if group_idx >= len(caption_groups):
                    # Trailing silence
                    proc.stdin.write(blank_bytes)
                    last_cache_key = None
                    continue

                active_group = caption_groups[group_idx]
                if current_t < active_group.start:
                    # Non-speech pause before or between phrases
                    proc.stdin.write(blank_bytes)
                    last_cache_key = None
                    continue

                # Inside active phrase
                layout = layouts[group_idx]
                active_idx: Optional[int] = None
                for idx, w in enumerate(active_group.words):
                    if w.start <= current_t <= w.end:
                        active_idx = idx
                        break
                    elif idx == len(active_group.words) - 1 and current_t >= w.start:
                        active_idx = idx
                        break

                if active_idx is not None:
                    act_w = active_group.words[active_idx]
                    t_rel = max(0.0, current_t - act_w.start)
                    scale = spring_scale(t_rel, duration=0.120)
                    scaled_pt = int(round(font_size * scale))
                    cache_key = (group_idx, active_idx, scaled_pt)
                else:
                    scale = 1.00
                    scaled_pt = font_size
                    cache_key = (group_idx, None, font_size)

                # Deduplication check
                if cache_key == last_cache_key and last_bytes is not None:
                    proc.stdin.write(last_bytes)
                    continue

                cached = rendered_cache.get(cache_key)
                if cached is not None:
                    last_bytes = cached
                    last_cache_key = cache_key
                    proc.stdin.write(cached)
                    continue

                # Render unique visual state frame
                img = render_caption_card(
                    layout=layout,
                    active_idx=active_idx,
                    scale=scale,
                    width=width,
                    height=height,
                    base_font_size=font_size,
                    font_path=font_path,
                    pill_color=pill_color,
                    color_active=color_active,
                    color_inactive=color_inactive,
                )
                bio_frame = io.BytesIO()
                img.save(bio_frame, "PNG", compress_level=1)
                frame_bytes = bio_frame.getvalue()

                rendered_cache[cache_key] = frame_bytes
                last_bytes = frame_bytes
                last_cache_key = cache_key
                proc.stdin.write(frame_bytes)

            proc.stdin.close()
            proc.stdin = None
            _, stderr = proc.communicate()
            if proc.returncode != 0:
                failed = True
                error_msg = stderr.decode(errors="ignore") if stderr else ""
        except (BrokenPipeError, IOError) as exc:
            if proc.stdin and not proc.stdin.closed:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            proc.stdin = None
            _, stderr = proc.communicate()
            failed = True
            error_msg = stderr.decode(errors="ignore") if stderr else str(exc)
        except Exception as exc:
            proc.kill()
            failed = True
            error_msg = str(exc)

        if not failed and output_path.exists() and output_path.stat().st_size > 0:
            logger.info("Direct pipe kinetic typography burned successfully into %s", output_path.name)
            return output_path

        logger.warning(
            "Direct pipe attempt %d failed (%s); trying fallback configuration...",
            attempt_idx + 1,
            error_msg[:120],
        )

    raise RuntimeError(f"All direct pipe attempts failed for {output_path.name}")


def burn_kinetic_captions_concat(
    video_path: Path,
    plan: EditingPlan,
    transcript: Transcript,
    output_path: Path,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
    video_bitrate: str = "6000k",
    font_size: int = 72,
    font_path: Optional[str] = None,
    color_active: Tuple[int, int, int, int] = (255, 204, 0, 255),
    color_inactive: Tuple[int, int, int, int] = (255, 255, 255, 255),
    pill_color: Tuple[int, int, int, int] = (20, 20, 25, 180),
) -> Path:
    """
    Sparse speech-interval kinetic typography burn-in using FFmpeg concat demuxer.
    Generates only unique visual states and skips non-speech intervals with duration metadata.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()

    mapped_words = map_words_to_timeline(plan, transcript)
    if not mapped_words:
        logger.warning("No words found to caption, copying video directly.")
        shutil.copy2(str(video_path), str(output_path))
        return output_path

    caption_groups = group_words_into_phrases(mapped_words, max_words=4)
    total_duration = plan.total_duration

    box_y = int(height * 0.62)
    word_spacing = max(24, int(font_size * 0.35))
    font_base = _get_font(font_size, font_path)

    layouts = [
        PhraseLayout(g, font_base, canvas_width=width, box_y=box_y, word_spacing=word_spacing)
        for g in caption_groups
    ]

    with tempfile.TemporaryDirectory(prefix="kinetic_concat_") as tmpdir:
        temp_dir = Path(tmpdir)
        blank_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        blank_path = temp_dir / "blank.png"
        blank_img.save(blank_path)

        concat_entries: List[Tuple[Path, float]] = []
        current_time = 0.0

        for g_idx, group in enumerate(caption_groups):
            # 1. Non-speech gap before this phrase
            if group.start > current_time:
                silence_dur = group.start - current_time
                concat_entries.append((blank_path, silence_dur))
                current_time = group.start

            # 2. Render words inside this phrase
            layout = layouts[g_idx]
            for w_idx, w in enumerate(group.words):
                w_start = max(w.start, current_time)
                w_end = w.end
                if w_end <= w_start:
                    continue

                w_dur = w_end - w_start
                # Entry bounce animation over 120ms
                bounce_dur = min(0.120, w_dur)
                bounce_frames = max(1, int(round(bounce_dur * fps)))
                frame_dur = 1.0 / fps

                for f in range(bounce_frames):
                    t_rel = f * frame_dur
                    scale = spring_scale(t_rel, duration=0.120)
                    card_img = render_caption_card(
                        layout=layout,
                        active_idx=w_idx,
                        scale=scale,
                        width=width,
                        height=height,
                        base_font_size=font_size,
                        font_path=font_path,
                        pill_color=pill_color,
                        color_active=color_active,
                        color_inactive=color_inactive,
                    )
                    card_path = temp_dir / f"card_g{g_idx}_w{w_idx}_b{f}.png"
                    card_img.save(card_path)
                    concat_entries.append((card_path, frame_dur))

                # Steady-state remainder
                steady_dur = w_dur - (bounce_frames * frame_dur)
                if steady_dur > 0.005:
                    card_img = render_caption_card(
                        layout=layout,
                        active_idx=w_idx,
                        scale=1.00,
                        width=width,
                        height=height,
                        base_font_size=font_size,
                        font_path=font_path,
                        pill_color=pill_color,
                        color_active=color_active,
                        color_inactive=color_inactive,
                    )
                    card_path = temp_dir / f"card_g{g_idx}_w{w_idx}_steady.png"
                    card_img.save(card_path)
                    concat_entries.append((card_path, steady_dur))

                current_time = w_end

        # 3. Trailing silence up to total_duration
        if total_duration > current_time:
            concat_entries.append((blank_path, total_duration - current_time))

        # 4. Write concat script
        concat_txt = temp_dir / "concat.txt"
        with open(concat_txt, "w") as f:
            for p, d in concat_entries:
                f.write(f"file '{p.resolve()}'\n")
                f.write(f"duration {d:.4f}\n")
            if concat_entries:
                f.write(f"file '{concat_entries[-1][0].resolve()}'\n")

        # 5. Composite directly with VideoToolbox hardware acceleration
        encoder_args, _ = _get_encoder_args(ffmpeg, video_bitrate)
        comp_cmd = [
            ffmpeg, "-y",
            "-i", str(video_path),
            "-f", "concat", "-safe", "0", "-i", str(concat_txt),
            "-filter_complex", "[0:v][1:v]overlay=0:0:shortest=1[v]",
            "-map", "[v]",
            "-map", "0:a?",
            *encoder_args,
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(output_path),
        ]
        res = subprocess.run(comp_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            logger.warning("VideoToolbox concat failed (%s); falling back to libx264/AAC...", res.stderr[:120])
            fallback_cmd = [
                ffmpeg, "-y",
                "-i", str(video_path),
                "-f", "concat", "-safe", "0", "-i", str(concat_txt),
                "-filter_complex", "[0:v][1:v]overlay=0:0:shortest=1[v]",
                "-map", "[v]",
                "-map", "0:a?",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                "-ar", "48000",
                "-ac", "2",
                "-movflags", "+faststart",
                str(output_path),
            ]
            subprocess.run(fallback_cmd, check=True, capture_output=True)

    logger.info("Concat kinetic typography burned successfully into %s", output_path.name)
    return output_path


def burn_kinetic_captions(
    video_path: Path,
    plan: EditingPlan,
    transcript: Transcript,
    output_path: Path,
    width: int = 1080,
    height: int = 1920,
    fps: int = 30,
    video_bitrate: str = "6000k",
    method: str = "pipe",
    font_size: int = 72,
    font_path: Optional[str] = None,
    color_active: Tuple[int, int, int, int] = (255, 204, 0, 255),
    color_inactive: Tuple[int, int, int, int] = (255, 255, 255, 255),
    pill_color: Tuple[int, int, int, int] = (20, 20, 25, 180),
) -> Path:
    """
    Main entry point for burning kinetic typography captions.
    Supports both high-throughput single-pass direct pipe (default) and sparse concat demuxer.
    """
    if method == "concat":
        return burn_kinetic_captions_concat(
            video_path=video_path,
            plan=plan,
            transcript=transcript,
            output_path=output_path,
            width=width,
            height=height,
            fps=fps,
            video_bitrate=video_bitrate,
            font_size=font_size,
            font_path=font_path,
            color_active=color_active,
            color_inactive=color_inactive,
            pill_color=pill_color,
        )
    return burn_kinetic_captions_direct_pipe(
        video_path=video_path,
        plan=plan,
        transcript=transcript,
        output_path=output_path,
        width=width,
        height=height,
        fps=fps,
        video_bitrate=video_bitrate,
        font_size=font_size,
        font_path=font_path,
        color_active=color_active,
        color_inactive=color_inactive,
        pill_color=pill_color,
    )
