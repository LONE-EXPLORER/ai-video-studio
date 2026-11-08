"""
scripts/generate_animated_map.py — Generates animated tactical/geopolitical map videos locally.

Produces a vertical 1080x1920 MP4 video clip featuring:
- High-contrast dark tactical cartography
- Highlighted territories & conflict zones
- Animated pan/zoom motion via FFmpeg
- Broadcast-grade lower-third title & coordinates
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

FFMPEG = "/opt/homebrew/bin/ffmpeg"


def create_tactical_map_frame(
    title: str = "PERSIAN GULF // STRATEGIC CHOKEPOINT",
    region: str = "STRAIT OF HORMUZ",
    output_png: Path = Path("/tmp/map_frame.png"),
    width: int = 1080,
    height: int = 1920,
) -> Path:
    """Create a high-resolution dark tactical map canvas."""
    img = Image.new("RGBA", (width, height), (10, 14, 20, 255))
    draw = ImageDraw.Draw(img)

    # Grid background (tactical radar style)
    grid_color = (25, 40, 60, 180)
    for x in range(0, width, 80):
        draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
    for y in range(0, height, 80):
        draw.line([(0, y), (width, y)], fill=grid_color, width=1)

    # Concentric radar circles
    center_x, center_y = width // 2, height // 2
    for r in [200, 350, 500]:
        draw.ellipse(
            [center_x - r, center_y - r, center_x + r, center_y + r],
            outline=(35, 65, 95, 160),
            width=2,
        )

    # Stylized tactical landmass polygon (Gulf & Strait representation)
    land_color = (20, 32, 48, 255)
    accent_red = (255, 60, 60, 220)
    accent_yellow = (255, 204, 0, 240)

    # Northern landmass (Iran / Mainland)
    draw.polygon([(0, 300), (450, 420), (650, 380), (1080, 500), (1080, 0), (0, 0)], fill=land_color, outline=(40, 80, 120, 255), width=3)
    
    # Southern landmass (Arabian Peninsula)
    draw.polygon([(0, 950), (400, 850), (600, 900), (900, 800), (1080, 850), (1080, 1920), (0, 1920)], fill=land_color, outline=(40, 80, 120, 255), width=3)

    # Strategic Chokepoint Pulse Circle (Strait of Hormuz)
    choke_x, choke_y = 540, 640
    draw.ellipse([choke_x - 90, choke_y - 90, choke_x + 90, choke_y + 90], outline=accent_red, width=4)
    draw.ellipse([choke_x - 40, choke_y - 40, choke_x + 40, choke_y + 40], fill=(255, 60, 60, 80), outline=accent_red, width=2)
    draw.line([(choke_x - 120, choke_y), (choke_x + 120, choke_y)], fill=accent_red, width=2)
    draw.line([(choke_x, choke_y - 120), (choke_x, choke_y + 120)], fill=accent_red, width=2)

    # Route Arrow (Chokepoint Flow)
    draw.line([(250, 680), (choke_x - 40, choke_y)], fill=accent_yellow, width=6)
    draw.line([(choke_x + 40, choke_y), (850, 600)], fill=accent_yellow, width=6)

    # Lower-third tactical HUD banner
    draw.rectangle([60, height - 380, width - 60, height - 160], fill=(5, 10, 18, 230), outline=(255, 204, 0, 255), width=2)
    
    # Typography
    font_large = ImageFont.load_default()
    draw.text((100, height - 340), "TACTICAL GEOPOLITICAL BRIEFING", fill=(255, 204, 0, 255), font=font_large)
    draw.text((100, height - 290), title, fill=(255, 255, 255, 255), font=font_large)
    draw.text((100, height - 240), f"ZONE: {region} // STATUS: CRITICAL", fill=(255, 80, 80, 255), font=font_large)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_png)
    return output_png


def render_animated_map_video(
    title: str,
    region: str,
    output_mp4: Path,
    duration: float = 3.0,
    width: int = 1080,
    height: int = 1920,
) -> Path:
    """Animate the tactical map frame using hardware-accelerated zoom/pan motion."""
    frame_png = Path("/tmp/tactical_map_frame.png")
    create_tactical_map_frame(title=title, region=region, output_png=frame_png, width=width, height=height)

    # Zoom-in effect over time
    vf_filter = (
        f"zoompan=z='min(zoom+0.0015,1.15)':d={int(duration*30)}:"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={width}x{height},"
        f"fps=30,setsar=1"
    )

    cmd = [
        FFMPEG, "-y",
        "-loop", "1", "-i", str(frame_png),
        "-vf", vf_filter,
        "-t", str(duration),
        "-c:v", "h264_videotoolbox",
        "-b:v", "5000k",
        "-pix_fmt", "nv12",
        "-movflags", "+faststart",
        str(output_mp4)
    ]
    subprocess.run(cmd, check=True)
    return output_mp4


if __name__ == "__main__":
    out = Path("assets/broll/geopolitics_animated_map.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    render_animated_map_video(
        title="THE 5-DAY OIL TRAP",
        region="STRAIT OF HORMUZ // PERSIAN GULF",
        output_mp4=out,
        duration=3.5,
    )
    print(f"Generated animated map: {out} ({out.stat().st_size / 1024:.1f} KB)")
