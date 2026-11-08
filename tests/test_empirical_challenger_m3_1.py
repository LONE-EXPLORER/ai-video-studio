"""
tests/test_empirical_challenger_m3_1.py — Empirical Challenger Adversarial Test Suite for Milestone 3.

Empirically challenges and stress-tests the video compositing and hardware synchronization
pipeline (`BrollCompositor`, Feature 11):
1. Aspect Ratio and Pixel SAR:
   - Feeds 16:9 landscape (1920x1080), 1:1 square (1080x1080), and 4:3 (1440x1080) B-roll inputs into BrollCompositor.
   - Probes output with ffprobe to assert resolution is exactly 1080x1920 and Pixel SAR is strictly 1:1.
   - Concatenates composited clip with unzoomed base clips via FFmpeg concat filter and concat_clips,
     asserting returncode 0 without crashes or SAR mismatch errors.
2. Dynamic PTS Progression (No Frozen Frame Bug):
   - Probes frames during the cutaway interval [timeline_start, timeline_end] using PIL and image diff.
   - Asserts pixel diff between frames within the cutaway is strictly greater than 0, proving frames animate.
   - Stress-tests looping progression where cutaway duration exceeds B-roll asset duration.
3. Master Dialogue Audio Integrity:
   - Verifies audio stream from track 0 has identical sample count and 48kHz sample rate before and after compositing.
   - Verifies zero audio leaks from B-roll (even if B-roll has an active audio stream) using FFT spectral analysis.
   - Verifies silent base video receives zero audio streams even when B-roll contains audio.
"""

from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
from PIL import Image, ImageChops

from app.rendering.broll_compositor import BrollCompositor
from app.schemas.editing_plan import BrollOverlay
from app.utils import ffmpeg_utils
from tests.e2e.conftest import make_synthetic_video


def _extract_frame_image(video_path: Path, timestamp: float, output_img: Path) -> Image.Image:
    """Extract a single frame from video at timestamp as a PIL Image."""
    ffmpeg = ffmpeg_utils.find_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-ss", f"{timestamp:.4f}",
        "-i", str(video_path),
        "-vframes", "1",
        str(output_img),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return Image.open(output_img).convert("RGB")


def _get_audio_sample_count_and_rate(video_path: Path, tmp_dir: Path) -> Tuple[int, int]:
    """Decode audio stream to raw PCM wav to count exact audio samples and sample rate."""
    ffmpeg = ffmpeg_utils.find_ffmpeg()
    wav_path = tmp_dir / f"audio_{video_path.stem}.wav"
    cmd = [
        ffmpeg, "-y",
        "-i", str(video_path),
        "-vn",
        "-c:a", "pcm_s16le",
        "-ar", "48000",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    with wave.open(str(wav_path), "rb") as wf:
        nframes = wf.getnframes()
        framerate = wf.getframerate()
    return nframes, framerate


def _extract_audio_spectrum(
    media_path: Path,
    start_sec: float,
    end_sec: float,
    tmp_dir: Path,
    target_sample_rate: int = 48000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract audio PCM slice and compute FFT magnitudes."""
    ffmpeg = ffmpeg_utils.find_ffmpeg()
    wav_path = tmp_dir / f"fft_{start_sec:.2f}_{end_sec:.2f}_{media_path.stem}.wav"
    cmd = [
        ffmpeg, "-y",
        "-ss", str(start_sec),
        "-to", str(end_sec),
        "-i", str(media_path),
        "-vn",
        "-c:a", "pcm_s16le",
        "-ar", str(target_sample_rate),
        "-ac", "1",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    with wave.open(str(wav_path), "rb") as wf:
        nframes = wf.getnframes()
        frames = wf.readframes(nframes)
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)

    fft_mag = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), 1.0 / target_sample_rate)
    return freqs, fft_mag


# ===========================================================================
# 1. Aspect Ratio and Pixel SAR Empirical Tests
# ===========================================================================

@pytest.mark.parametrize(
    "width,height,label",
    [
        (1920, 1080, "16x9_landscape"),
        (1080, 1080, "1x1_square"),
        (1440, 1080, "4x3_standard"),
    ],
)
@pytest.mark.parametrize("use_videotoolbox", [True, False])
def test_empirical_broll_aspect_ratio_and_pixel_sar(
    tmp_path: Path, width: int, height: int, label: str, use_videotoolbox: bool
):
    """
    Empirical Assertion:
    Feed 16:9 landscape (1920x1080), 1:1 square (1080x1080), and 4:3 (1440x1080) B-roll inputs
    into BrollCompositor. Probing with ffprobe must assert:
    - Width is exactly 1080, Height is exactly 1920.
    - Pixel SAR is strictly 1:1 ('1:1' or '1/1').
    - DAR is strictly 9:16 ('9:16' or equivalent).
    - Tests under both VideoToolbox hardware encoding and software libx264 fallback.
    """
    base = tmp_path / f"base_{label}_{use_videotoolbox}.mp4"
    broll = tmp_path / f"broll_{label}_{use_videotoolbox}.mp4"
    out = tmp_path / f"composited_{label}_{use_videotoolbox}.mp4"

    make_synthetic_video(base, width=1080, height=1920, fps=30, duration=2.5, has_audio=True)
    make_synthetic_video(broll, width=width, height=height, fps=30, duration=1.5, pattern="testsrc")

    compositor = BrollCompositor()
    compositor._vt_available = use_videotoolbox

    overlay = BrollOverlay(
        broll_id=f"broll_{label}",
        timeline_start=0.5,
        timeline_end=2.0,
        source_path=str(broll),
        overlay_type="cutaway",
    )

    result = compositor.composite_overlays(
        base_video_path=base,
        overlays=[overlay],
        output_path=out,
        target_w=1080,
        target_h=1920,
        fps=30,
    )

    assert result.exists()
    assert result.stat().st_size > 0

    info = ffmpeg_utils.probe_video(result)
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")

    # Strict resolution check
    assert vstream["width"] == 1080, f"Expected 1080 width, got {vstream['width']} for {label}"
    assert vstream["height"] == 1920, f"Expected 1920 height, got {vstream['height']} for {label}"

    # Strict Pixel SAR check: must be 1:1
    sar = vstream.get("sample_aspect_ratio", "1:1")
    assert sar in ("1:1", "1/1"), f"Invalid Pixel SAR '{sar}' for {label} (vt={use_videotoolbox})"

    # Display Aspect Ratio check
    dar = vstream.get("display_aspect_ratio", "9:16")
    assert dar in ("9:16", None), f"Invalid DAR '{dar}' for {label} (vt={use_videotoolbox})"


@pytest.mark.parametrize(
    "width,height,label",
    [
        (1920, 1080, "16x9_landscape"),
        (1080, 1080, "1x1_square"),
        (1440, 1080, "4x3_standard"),
    ],
)
def test_empirical_concat_stability_with_unzoomed_base_clips(
    tmp_path: Path, width: int, height: int, label: str
):
    """
    Empirical Assertion:
    Concatenating the composited clip (originating from 16:9, 1:1, or 4:3 B-roll inputs)
    with unzoomed base clips must not crash FFmpeg concat filter or concat_clips.
    Verifies filter_complex concat executes with returncode 0 and preserves SAR 1:1.
    """
    base = tmp_path / f"base_{label}.mp4"
    unzoomed_head = tmp_path / f"unzoomed_head_{label}.mp4"
    unzoomed_tail = tmp_path / f"unzoomed_tail_{label}.mp4"
    broll = tmp_path / f"broll_{label}.mp4"
    composited = tmp_path / f"composited_{label}.mp4"
    concat_out = tmp_path / f"concat_{label}.mp4"

    make_synthetic_video(base, width=1080, height=1920, fps=30, duration=2.0, has_audio=True, audio_freq=440)
    make_synthetic_video(unzoomed_head, width=1080, height=1920, fps=30, duration=1.5, has_audio=True, audio_freq=440)
    make_synthetic_video(unzoomed_tail, width=1080, height=1920, fps=30, duration=1.5, has_audio=True, audio_freq=440)
    make_synthetic_video(broll, width=width, height=height, fps=30, duration=1.0, pattern="testsrc")

    compositor = BrollCompositor()
    overlay = BrollOverlay(
        broll_id=f"broll_{label}",
        timeline_start=0.5,
        timeline_end=1.5,
        source_path=str(broll),
        overlay_type="cutaway",
    )
    compositor.composite_overlays(base, [overlay], composited)

    # 1. Test concat via ffmpeg_utils.concat_clips
    res = ffmpeg_utils.concat_clips([unzoomed_head, composited, unzoomed_tail], concat_out)
    assert res.exists()
    assert res.stat().st_size > 0

    # 2. Test direct FFmpeg filter_complex concat to assert strict exit code 0
    ffmpeg = ffmpeg_utils.find_ffmpeg()
    direct_out = tmp_path / f"direct_concat_{label}.mp4"
    cmd = [
        ffmpeg, "-y",
        "-i", str(unzoomed_head),
        "-i", str(composited),
        "-i", str(unzoomed_tail),
        "-filter_complex", "[0:v][0:a][1:v][1:a][2:v][2:a]concat=n=3:v=1:a=1[outv][outa]",
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-c:a", "aac",
        str(direct_out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"FFmpeg concat crashed for {label}! Stderr: {proc.stderr}"

    # Verify concatenated output SAR & geometry
    info = ffmpeg_utils.probe_video(direct_out)
    vs = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert vs["width"] == 1080
    assert vs["height"] == 1920
    assert vs.get("sample_aspect_ratio", "1:1") in ("1:1", "1/1")
    assert vs.get("display_aspect_ratio", "9:16") in ("9:16", None)


# ===========================================================================
# 2. Dynamic PTS Progression (No Frozen Frame Bug) Tests
# ===========================================================================

def test_empirical_dynamic_pts_progression_within_cutaway(tmp_path: Path):
    """
    Empirical Assertion:
    Probe frames during the cutaway interval [timeline_start, timeline_end] using PIL.
    Assert that pixel diff between frames within the cutaway is strictly greater than 0,
    proving that B-roll frames actively animate and are NOT frozen.
    """
    base = tmp_path / "base_dynamic.mp4"
    broll = tmp_path / "broll_dynamic.mp4"
    composited = tmp_path / "composited_dynamic.mp4"

    # Base video with static color bars
    make_synthetic_video(base, width=1080, height=1920, fps=30, duration=4.0, pattern="smptebars")
    # Dynamic B-roll video where every frame actively animates (testsrc has running timer & moving shapes)
    make_synthetic_video(broll, width=1080, height=1920, fps=30, duration=2.0, pattern="testsrc")

    timeline_start = 1.0
    timeline_end = 3.0

    compositor = BrollCompositor()
    overlay = BrollOverlay(
        broll_id="broll_anim",
        timeline_start=timeline_start,
        timeline_end=timeline_end,
        source_path=str(broll),
        overlay_type="cutaway",
    )
    compositor.composite_overlays(base, [overlay], composited)

    # Extract frames across timeline:
    # Frame 0: t=0.5s (before cutaway, base video)
    # Frame 1: t=1.2s (early cutaway, B-roll)
    # Frame 2: t=1.8s (mid cutaway, B-roll)
    # Frame 3: t=2.5s (late cutaway, B-roll)
    # Frame 4: t=3.5s (after cutaway, base video)
    f_pre = _extract_frame_image(composited, 0.5, tmp_path / "f_pre.png")
    f_cut1 = _extract_frame_image(composited, 1.2, tmp_path / "f_cut1.png")
    f_cut2 = _extract_frame_image(composited, 1.8, tmp_path / "f_cut2.png")
    f_cut3 = _extract_frame_image(composited, 2.5, tmp_path / "f_cut3.png")
    f_post = _extract_frame_image(composited, 3.5, tmp_path / "f_post.png")

    # 1. Assert animation WITHIN the cutaway interval [timeline_start, timeline_end]
    diff_cut1_cut2 = ImageChops.difference(f_cut1, f_cut2)
    bbox_cut1_cut2 = diff_cut1_cut2.getbbox()
    mean_diff_1_2 = float(np.mean(np.abs(np.array(f_cut1, dtype=float) - np.array(f_cut2, dtype=float))))

    assert bbox_cut1_cut2 is not None, "Cutaway frames at t=1.2s and t=1.8s are identical (FROZEN FRAME BUG)!"
    assert mean_diff_1_2 > 0.5, f"Mean pixel diff {mean_diff_1_2} <= 0.5; frames are practically static!"

    diff_cut2_cut3 = ImageChops.difference(f_cut2, f_cut3)
    bbox_cut2_cut3 = diff_cut2_cut3.getbbox()
    mean_diff_2_3 = float(np.mean(np.abs(np.array(f_cut2, dtype=float) - np.array(f_cut3, dtype=float))))

    assert bbox_cut2_cut3 is not None, "Cutaway frames at t=1.8s and t=2.5s are identical (FROZEN FRAME BUG)!"
    assert mean_diff_2_3 > 0.5, f"Mean pixel diff {mean_diff_2_3} <= 0.5; frames are practically static!"

    # 2. Assert transition boundaries: base -> B-roll and B-roll -> base
    diff_pre_cut = ImageChops.difference(f_pre, f_cut1)
    assert diff_pre_cut.getbbox() is not None, "Pre-cutaway base and cutaway frame are identical!"

    diff_cut_post = ImageChops.difference(f_cut3, f_post)
    assert diff_cut_post.getbbox() is not None, "Cutaway frame and post-cutaway base are identical!"


def test_empirical_dynamic_pts_looping_when_cutaway_exceeds_asset(tmp_path: Path):
    """
    Empirical Assertion:
    Stress-test looping progression when cutaway duration (3.0s) exceeds asset duration (0.8s).
    The looped B-roll frames must continue to animate dynamically past the asset duration
    without freezing or hanging on the last frame.
    """
    base = tmp_path / "base_loop.mp4"
    broll_short = tmp_path / "broll_short.mp4"
    composited = tmp_path / "composited_loop.mp4"

    # Base 5.0s, B-roll asset is only 0.8s short
    make_synthetic_video(base, width=1080, height=1920, fps=30, duration=5.0, pattern="smptebars")
    make_synthetic_video(broll_short, width=1080, height=1920, fps=30, duration=0.8, pattern="testsrc")

    compositor = BrollCompositor()
    overlay = BrollOverlay(
        broll_id="broll_short",
        timeline_start=1.0,
        timeline_end=4.0,  # 3.0s cutaway >> 0.8s asset
        source_path=str(broll_short),
        overlay_type="cutaway",
    )
    compositor.composite_overlays(base, [overlay], composited)

    # Extract frames during loop iterations:
    # t=1.5s (iteration 1, offset 0.5s)
    # t=2.3s (iteration 2, offset 0.5s)
    # t=3.6s (iteration 4, offset 0.2s)
    f_loop1 = _extract_frame_image(composited, 1.5, tmp_path / "f_l1.png")
    f_loop2 = _extract_frame_image(composited, 2.3, tmp_path / "f_l2.png")
    f_loop4 = _extract_frame_image(composited, 3.6, tmp_path / "f_l4.png")

    # Frames at t=2.3s and t=3.6s are well beyond the original 0.8s asset duration.
    # Assert they are animating and not frozen.
    diff_looped = ImageChops.difference(f_loop2, f_loop4)
    bbox_looped = diff_looped.getbbox()
    mean_diff_looped = float(np.mean(np.abs(np.array(f_loop2, dtype=float) - np.array(f_loop4, dtype=float))))

    assert bbox_looped is not None, "Looped B-roll frames froze after original asset duration!"
    assert mean_diff_looped > 0.5, f"Mean pixel diff across loops {mean_diff_looped} <= 0.5; video froze!"


# ===========================================================================
# 3. Master Dialogue Audio Integrity Tests
# ===========================================================================

def test_empirical_master_dialogue_audio_sample_count_and_rate(tmp_path: Path):
    """
    Empirical Assertion:
    Verify audio stream from track 0 has identical sample count and 48kHz sample rate
    before and after B-roll compositing.
    """
    base = tmp_path / "base_master_audio.mp4"
    broll = tmp_path / "broll_noisy_audio.mp4"
    composited = tmp_path / "composited_master_audio.mp4"

    # Base video with 3.5s 48kHz stereo master dialogue audio
    make_synthetic_video(base, width=1080, height=1920, fps=30, duration=3.5, has_audio=True, audio_freq=440)
    # B-roll with distinct audio frequency
    make_synthetic_video(broll, width=1080, height=1920, fps=30, duration=2.0, has_audio=True, audio_freq=990)

    compositor = BrollCompositor()
    overlay = BrollOverlay(
        broll_id="broll_audit",
        timeline_start=0.8,
        timeline_end=2.6,
        source_path=str(broll),
        overlay_type="cutaway",
    )
    compositor.composite_overlays(base, [overlay], composited)

    # 1. Exact sample count & sample rate audit via PCM decoding
    base_samples, base_rate = _get_audio_sample_count_and_rate(base, tmp_path)
    comp_samples, comp_rate = _get_audio_sample_count_and_rate(composited, tmp_path)

    assert comp_rate == 48000, f"Expected 48000Hz, got {comp_rate}Hz"
    assert base_rate == 48000, f"Base was not 48000Hz: {base_rate}Hz"
    assert comp_samples == base_samples, (
        f"Audio sample count mismatch! Before: {base_samples}, After: {comp_samples} "
        f"(diff: {comp_samples - base_samples} samples)"
    )

    # 2. FFprobe packet and stream count audit
    cmd_packets = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=nb_read_packets,sample_rate,channels",
        "-count_packets",
        "-of", "json",
        str(composited),
    ]
    res = subprocess.run(cmd_packets, capture_output=True, text=True, check=True)
    probe_data = json.loads(res.stdout)

    streams = probe_data.get("streams", [])
    assert len(streams) == 1, f"Expected exactly 1 audio stream, found {len(streams)}"
    assert streams[0]["sample_rate"] == "48000"
    assert streams[0]["channels"] == 2


def test_empirical_zero_audio_leak_spectral_isolation(tmp_path: Path):
    """
    Empirical Assertion:
    Verify zero audio leaks from B-roll (even if B-roll input has an active, loud audio stream).
    Base video has 440Hz tone; B-roll has loud 2500Hz tone.
    FFT spectral analysis during cutaway must confirm 440Hz dialogue is preserved,
    and 2500Hz B-roll tone is completely absent (< -80dB / noise floor).
    """
    base = tmp_path / "base_dialogue.mp4"
    broll_loud = tmp_path / "broll_loud.mp4"
    composited = tmp_path / "composited_spectral.mp4"

    # Base video: 440Hz dialogue tone
    make_synthetic_video(base, width=1080, height=1920, fps=30, duration=4.0, has_audio=True, audio_freq=440)
    # B-roll: 2500Hz tone
    make_synthetic_video(broll_loud, width=1080, height=1920, fps=30, duration=2.5, has_audio=True, audio_freq=2500)

    compositor = BrollCompositor()
    overlay = BrollOverlay(
        broll_id="broll_leak_test",
        timeline_start=1.0,
        timeline_end=3.0,
        source_path=str(broll_loud),
        overlay_type="cutaway",
    )
    compositor.composite_overlays(base, [overlay], composited)

    # Perform FFT spectral analysis during the cutaway interval (1.2s to 2.8s)
    freqs, fft_mag = _extract_audio_spectrum(composited, 1.2, 2.8, tmp_path, target_sample_rate=48000)

    idx_440 = int(np.argmin(np.abs(freqs - 440.0)))
    idx_2500 = int(np.argmin(np.abs(freqs - 2500.0)))

    power_440 = float(fft_mag[idx_440])
    power_2500 = float(fft_mag[idx_2500])

    # 440Hz dialogue must be dominant
    assert power_440 > 10000.0, f"Master dialogue 440Hz audio missing or muted! Power={power_440}"

    # 2500Hz B-roll tone must not leak into master dialogue
    leak_ratio = power_2500 / max(power_440, 1e-6)
    assert leak_ratio < 1e-4, (
        f"B-roll audio leaked into master dialogue track! "
        f"Power@2500Hz={power_2500:.2f}, Power@440Hz={power_440:.2f}, LeakRatio={leak_ratio:.2e}"
    )


def test_empirical_silent_base_zero_audio_leak(tmp_path: Path):
    """
    Empirical Assertion:
    If base video has no audio track, but B-roll input HAS an audio track,
    BrollCompositor must output a video with ZERO audio streams (-an),
    guaranteeing no unwanted secondary audio is introduced.
    """
    base_silent = tmp_path / "base_silent.mp4"
    broll_loud = tmp_path / "broll_loud_silent_test.mp4"
    composited = tmp_path / "composited_silent.mp4"

    make_synthetic_video(base_silent, width=1080, height=1920, fps=30, duration=2.5, has_audio=False)
    make_synthetic_video(broll_loud, width=1080, height=1920, fps=30, duration=1.5, has_audio=True, audio_freq=880)

    compositor = BrollCompositor()
    overlay = BrollOverlay(
        broll_id="broll_silent_audit",
        timeline_start=0.5,
        timeline_end=2.0,
        source_path=str(broll_loud),
        overlay_type="cutaway",
    )
    compositor.composite_overlays(base_silent, [overlay], composited)

    info = ffmpeg_utils.probe_video(composited)
    astreams = [s for s in info["streams"] if s["codec_type"] == "audio"]
    assert len(astreams) == 0, f"Expected 0 audio streams, found {len(astreams)}: B-roll audio leaked into silent base!"
