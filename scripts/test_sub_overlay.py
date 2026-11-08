"""
Test subtitle video generation with alpha channel.
"""
import time
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def test():
    t0 = time.time()
    w, h = 1080, 260
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Impact.ttf", 68)
    
    # We will generate a 3-second transparent WebM (or ProRes 4444) at 10 fps
    # Using ffmpeg image2pipe with PNG
    cmd = [
        "/opt/homebrew/bin/ffmpeg", "-y",
        "-f", "image2pipe",
        "-framerate", "10",
        "-i", "-",
        "-c:v", "png",
        "jobs/test_subs.mov"
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    words = ["STOP", "EVERYTHING,", "FORGET", "THE", "HEADLINES"]
    for i in range(30):
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        active_idx = (i // 6) % len(words)
        
        # Calculate positions
        text = " ".join(words)
        # Draw words with active highlight
        cur_x = 50
        y = 100
        for idx, word in enumerate(words):
            color = (255, 204, 0, 255) if idx == active_idx else (255, 255, 255, 255)
            draw.text((cur_x + 3, y + 3), word, font=font, fill=(0, 0, 0, 200))
            draw.text((cur_x, y), word, font=font, fill=color, stroke_width=5, stroke_fill=(0, 0, 0, 255))
            bbox = draw.textbbox((cur_x, y), word, font=font)
            cur_x += (bbox[2] - bbox[0]) + 20
            
        img.save(proc.stdin, "PNG")
        
    proc.stdin.close()
    proc.wait()
    dt = time.time() - t0
    out = Path("jobs/test_subs.mov")
    print(f"Generated transparent subtitle video in {dt:.2f}s: {out} (size={out.stat().st_size} bytes)")
    
    # Now test overlay onto a snippet
    t1 = time.time()
    overlay_cmd = [
        "/opt/homebrew/bin/ffmpeg", "-y",
        "-ss", "0", "-t", "3",
        "-i", "assets/test_videos/trump_5_day_trap.mp4",
        "-i", "jobs/test_subs.mov",
        "-filter_complex", "[0:v][1:v]overlay=(W-w)/2:H-h-200:shortest=1",
        "-c:v", "libx264", "-c:a", "copy",
        "jobs/test_overlay_result.mp4"
    ]
    res = subprocess.run(overlay_cmd, capture_output=True, text=True)
    dt2 = time.time() - t1
    res_out = Path("jobs/test_overlay_result.mp4")
    print(f"Overlay complete in {dt2:.2f}s (code={res.returncode}): {res_out} (size={res_out.stat().st_size if res_out.exists() else 0} bytes)")
    if res.returncode != 0:
        print("STDERR:", res.stderr[-500:])

if __name__ == "__main__":
    test()
