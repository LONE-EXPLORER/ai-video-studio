"""
Test script to verify Pillow kinetic caption overlay on macOS.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import subprocess

def test():
    font_path = "/System/Library/Fonts/Supplemental/Impact.ttf"
    font = ImageFont.truetype(font_path, 72)
    
    img = Image.new("RGBA", (1080, 1920), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    text = "STOP EVERYTHING"
    # Draw with black outline and yellow fill
    bbox = draw.textbbox((540, 1150), text, font=font, anchor="mm")
    
    # Shadow
    draw.text((544, 1154), text, font=font, fill=(0, 0, 0, 220), anchor="mm")
    # Stroke
    draw.text((540, 1150), text, font=font, fill=(255, 204, 0, 255), anchor="mm", stroke_width=6, stroke_fill=(0, 0, 0, 255))
    
    out_png = Path("jobs/test_caption.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    print(f"Saved test caption PNG to {out_png} (size={out_png.stat().st_size} bytes)")

if __name__ == "__main__":
    test()
