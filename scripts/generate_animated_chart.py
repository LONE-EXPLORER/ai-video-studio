"""
scripts/generate_animated_chart.py — Generates animated financial/economic chart videos locally.

Produces a vertical 1080x1920 MP4 video clip featuring:
- High-contrast glowing neon chart line
- Animated progressive line draw & data point reveal
- Giant metric counter with percentage callout
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

FFMPEG = "/opt/homebrew/bin/ffmpeg"


def render_animated_chart_video(
    metric_title: str,
    data_labels: list[str],
    data_values: list[float],
    callout: str,
    output_mp4: Path,
    duration: float = 3.0,
    width: int = 1080,
    height: int = 1920,
) -> Path:
    """Render a progressive neon financial chart video frame-by-frame via FFmpeg pipe."""
    fps = 30
    total_frames = int(duration * fps)

    # Normalize values for plotting
    min_v, max_v = min(data_values) * 0.9, max(data_values) * 1.15
    val_range = max(max_v - min_v, 1.0)

    plot_left, plot_right = 140, width - 140
    plot_bottom, plot_top = height - 500, height - 1200
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    points = []
    n_pts = len(data_values)
    for i, v in enumerate(data_values):
        px = plot_left + int(i * (plot_width / max(n_pts - 1, 1)))
        py = plot_bottom - int(((v - min_v) / val_range) * plot_height)
        points.append((px, py))

    cmd = [
        FFMPEG, "-y",
        "-f", "image2pipe",
        "-framerate", str(fps),
        "-i", "-",
        "-t", str(duration),
        "-c:v", "h264_videotoolbox",
        "-b:v", "6000k",
        "-pix_fmt", "nv12",
        "-movflags", "+faststart",
        str(output_mp4)
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    font = ImageFont.load_default()

    for frame_idx in range(total_frames):
        progress = min(1.0, frame_idx / (total_frames * 0.75))  # Finish draw by 75% duration

        img = Image.new("RGBA", (width, height), (12, 16, 24, 255))
        draw = ImageDraw.Draw(img)

        # Background grid
        for gy in range(plot_top, plot_bottom + 1, 100):
            draw.line([(plot_left, gy), (plot_right, gy)], fill=(30, 42, 60, 200), width=1)

        # Draw Title HUD
        draw.rectangle([80, 250, width - 80, 480], fill=(18, 24, 36, 240), outline=(0, 230, 153, 255), width=2)
        draw.text((120, 280), "FINANCIAL METRIC TRACKER", fill=(0, 230, 153, 255), font=font)
        draw.text((120, 330), metric_title, fill=(255, 255, 255, 255), font=font)
        draw.text((120, 390), f"IMPACT: {callout}", fill=(255, 204, 0, 255), font=font)

        # Progressive chart line
        max_idx = int(progress * (len(points) - 1))
        drawn_pts = points[:max_idx + 1]

        if len(drawn_pts) >= 2:
            # Glow effect (thick translucent lines)
            draw.line(drawn_pts, fill=(0, 230, 153, 100), width=16)
            draw.line(drawn_pts, fill=(0, 230, 153, 180), width=8)
            draw.line(drawn_pts, fill=(255, 255, 255, 255), width=4)

        for pt in drawn_pts:
            draw.ellipse([pt[0] - 8, pt[1] - 8, pt[0] + 8, pt[1] + 8], fill=(255, 204, 0, 255), outline=(255, 255, 255, 255), width=2)

        # X-Axis labels
        for i, lbl in enumerate(data_labels):
            lx = plot_left + int(i * (plot_width / max(n_pts - 1, 1)))
            draw.text((lx - 20, plot_bottom + 30), lbl, fill=(160, 180, 200, 255), font=font)

        img.convert("RGB").save(proc.stdin, "JPEG")

    proc.stdin.close()
    proc.wait()
    return output_mp4


if __name__ == "__main__":
    out = Path("assets/broll/crude_oil_surge_chart.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    render_animated_chart_video(
        metric_title="GLOBAL CRUDE OIL SURGE ($/BARREL)",
        data_labels=["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"],
        data_values=[74.0, 82.0, 96.5, 118.0, 142.5],
        callout="+92.5% EXPLOSION",
        output_mp4=out,
        duration=3.5,
    )
    print(f"Generated animated chart: {out} ({out.stat().st_size / 1024:.1f} KB)")
