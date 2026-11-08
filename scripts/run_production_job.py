"""
scripts/run_production_job.py — Executes an end-to-end job on the user's video and verifies all deliverables.
"""

import sys
import time
import uuid
from pathlib import Path

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from app.config import get_config
from app.providers.ollama_provider import OllamaProvider
from app.pipeline.orchestrator import PipelineOrchestrator


def main():
    video_source = Path("/Users/pranjalshahi/.gemini/antigravity/brain/cda086f5-a17a-4d2a-8fba-ab0f76f5d283/.user_uploaded/media_1793829730655_4e653906.mp4")
    if not video_source.exists():
        video_source = _ROOT / "assets" / "test_videos" / "trump_5_day_trap.mp4"
    
    print(f"Using source video: {video_source} (exists={video_source.exists()})")

    cfg = get_config()
    provider = OllamaProvider(
        host=cfg.ollama_host,
        model=cfg.ollama_model,
        fallback_model=cfg.ollama_fallback_model,
    )

    orchestrator = PipelineOrchestrator(provider=provider)

    job_id = f"prod-{uuid.uuid4()}"
    prompt = (
        "You are an expert, high-retention short-form video editor. "
        "I need you to write the necessary Python scripts using ffmpeg, auto-editor, and mlx-whisper to process my video named _Trump’s 5-Day Trap. "
        "Execute this 4-step pipeline and output a DaVinci Resolve timeline: "
        "Step 1: Aggressive Pacing (The Cut) Use mlx-whisper to generate a word-level transcript. "
        "Then, use auto-editor to remove all silences longer than 0.15 seconds. The pacing must be relentless. "
        "Step 2: Cinematic Motion (The 'Punch-In') I want dynamic framing. Write the code so that for the first 3 seconds "
        "('Stop everything, forget the headlines'), the video is zoomed in at 115% scale. "
        "After that, for every single jump cut created by auto-editor, alternate the video scale between 100% and 108%. "
        "Step 3: Kinetic Typography (The Captions) Generate an .ass subtitle file with karaoke effect in Bright Yellow (#FFCC00), "
        "white text, heavy black drop shadow. Step 4: Final Output Combine edits and export .fcpxml timeline."
    )

    print(f"\n=======================================================")
    print(f"🚀 Starting Production Pipeline Job: {job_id}")
    print(f"Profile: dynamic_shorts | Aspect: 9:16 | Target: 60s")
    print(f"Music: True | Model: {cfg.ollama_model}")
    print(f"=======================================================\n")

    def progress_callback(stage: str, status: str, message: str):
        print(f"[{time.strftime('%X')}] [{stage.upper():10s}] [{status.upper():7s}] {message}")

    start_t = time.time()
    results = orchestrator.run_job(
        job_id=job_id,
        source_video_path=video_source,
        user_instruction=prompt,
        profile="dynamic_shorts",
        target_duration=60.0,
        aspect_ratio="9:16",
        caption_style="dynamic",
        enable_music=True,
        progress_callback=progress_callback,
    )
    total_time = time.time() - start_t

    print(f"\n=======================================================")
    print(f"🎉 PIPELINE EXECUTION COMPLETED IN {total_time:.1f}s!")
    print(f"=======================================================")
    
    final_video = results.get("final_video")
    print(f"Final MP4 Video:       {final_video} (exists={final_video.exists() if final_video else False})")
    if final_video and final_video.exists():
        print(f"Final Video Size:      {final_video.stat().st_size / (1024*1024):.2f} MB")
        
    fcpxml = results.get("fcpxml_path")
    print(f"DaVinci FCPXML:        {fcpxml} (exists={fcpxml.exists() if fcpxml else False})")
    
    plan = results.get("plan")
    if plan:
        print(f"Total Clips Cut:       {len(plan.clips)}")
        print(f"Hook Summary:          {plan.hook_summary}")
        print(f"Total Plan Duration:   {plan.total_duration:.2f}s")
        
    preview = results.get("preview_video")
    print(f"Web Preview Video:     {preview} (exists={preview.exists() if preview else False})")
    
    quality = results.get("quality_report")
    if quality:
        failed = [c.name for c in quality.failed_checks]
        print(f"Quality Check Status:  Passed={quality.passed} (failed checks: {failed})")

    # Copy deliverables to canonical deliverables/ directory
    import shutil
    deliv_dir = _ROOT / "deliverables"
    deliv_dir.mkdir(parents=True, exist_ok=True)
    if final_video and final_video.exists():
        master_mp4 = deliv_dir / "Trump_5DayTrap_Marvel_Master.mp4"
        shutil.copy2(final_video, master_mp4)
        print(f"\n✨ Canonical Deliverable Saved: {master_mp4} ({master_mp4.stat().st_size / (1024*1024):.2f} MB)")
    if fcpxml and fcpxml.exists():
        master_fcpxml = deliv_dir / "Trump_5DayTrap_Marvel_Timeline.fcpxml"
        shutil.copy2(fcpxml, master_fcpxml)
        print(f"✨ Canonical Timeline Saved:    {master_fcpxml}")


if __name__ == "__main__":
    main()
