"""
app/ui/components.py — Shared Streamlit UI components and widgets with time-weighted progress tracking.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import streamlit as st

# Realistic time-weight distribution based on actual execution profiles on Apple Silicon
STAGE_WEIGHTS: Dict[str, float] = {
    "ingest": 0.03,      # 3% - Quick container probe, metadata & copy (~0.5 - 1s)
    "transcribe": 0.22,  # 22% - Audio extraction & MLX Whisper speech-to-text (~5 - 8s)
    "analyze": 0.05,     # 5% - Keyframe extraction & scene detection (~1 - 2s)
    "plan": 0.25,        # 25% - LLM reasoning & structured JSON cut plan (~8 - 15s)
    "edit": 0.15,        # 15% - Subprocess clip cuts, zooms, concat demuxer (~3 - 5s)
    "render": 0.22,      # 22% - Final H.264 VideoToolbox render & loudnorm (~6 - 10s)
    "quality": 0.03,     # 3% - Automated verification checks (~0.5s)
    "preview": 0.05,     # 5% - Fast web-optimized preview encode (~1.5s)
}


def render_stage_progress(
    stages: List[Dict[str, Any]],
    start_time: Optional[float] = None,
) -> None:
    """
    Render a mathematically accurate, time-weighted progress bar with live timing and ETA.
    """
    icon_map = {
        "pending": "⏳",
        "running": "🔄",
        "done": "✅",
        "failed": "❌",
        "skipped": "⏭️",
    }
    
    # Calculate time-weighted completion percentage
    accumulated_pct = 0.0
    current_running_stage = None
    failed_stage = None
    
    for s in stages:
        s_name = s.get("name", "").lower()
        weight = STAGE_WEIGHTS.get(s_name, 1.0 / len(stages))
        status = s.get("status", "pending")
        
        if status == "done":
            accumulated_pct += weight
        elif status == "running":
            current_running_stage = s
            # Start very small when running (15% of this stage's allocated weight), not jumping halfway
            accumulated_pct += (weight * 0.15)
        elif status == "failed":
            failed_stage = s
            break

    clamped_pct = min(1.0, max(0.0, accumulated_pct))
    pct_display = int(round(clamped_pct * 100))

    # Elapsed time & ETA calculation
    timing_text = ""
    if start_time is not None:
        elapsed = time.time() - start_time
        if clamped_pct > 0.05 and clamped_pct < 0.99:
            estimated_total = elapsed / clamped_pct
            eta = max(0.0, estimated_total - elapsed)
            timing_text = f"⏱️ Elapsed: {elapsed:.1f}s • Estimated Remaining: ~{eta:.0f}s"
        else:
            timing_text = f"⏱️ Elapsed: {elapsed:.1f}s"

    # Top Hero Progress Header
    if failed_stage:
        st.markdown(f"### ❌ Pipeline Failed at **{pct_display}%** ({failed_stage.get('name', '').capitalize()})")
        st.progress(clamped_pct, text=f"Stopped at {pct_display}% • {timing_text}")
    elif clamped_pct >= 0.99 or all(s.get("status") == "done" for s in stages):
        st.markdown("### 🎉 Overall Completion: **100%**")
        st.progress(1.0, text=f"All stages completed successfully! • {timing_text}")
    else:
        current_label = current_running_stage.get("name", "").capitalize() if current_running_stage else "Starting"
        st.markdown(f"### 📈 Overall Completion: **{pct_display}%** ({current_label})")
        st.progress(clamped_pct, text=f"Progress: {pct_display}% • {timing_text}")

    st.markdown("---")
    st.markdown("#### Real-time Stage Breakdown (Weighted by Processing Time)")

    # Render each stage row with its true time-weight
    cumulative_target = 0.0
    for i, s in enumerate(stages):
        step_num = i + 1
        s_name = s.get("name", "").lower()
        name_display = s.get("name", "").capitalize()
        status = s.get("status", "pending")
        msg = s.get("message", "")
        icon = icon_map.get(status, "⏳")
        
        weight = STAGE_WEIGHTS.get(s_name, 1.0 / len(stages))
        cumulative_target += weight
        stage_target_pct = int(round(cumulative_target * 100))
        weight_pct = int(round(weight * 100))
        
        col_icon, col_step, col_name, col_bar, col_status = st.columns([0.8, 1.4, 2.8, 3.2, 4.0])
        
        with col_icon:
            st.markdown(f"### {icon}")
        with col_step:
            st.markdown(f"**Step {step_num}/8**")
            st.caption(f"Weight: {weight_pct}%")
        with col_name:
            st.markdown(f"**{name_display}**")
            st.caption(f"Milestone: {stage_target_pct}%")
        with col_bar:
            if status == "done":
                st.progress(1.0, text=f"✅ Done (100%)")
            elif status == "running":
                # When running, start small (15%) rather than jumping to 50%
                st.progress(0.15, text=f"🔄 In progress...")
            elif status == "failed":
                st.progress(1.0, text=f"❌ Failed")
            else:
                st.progress(0.0, text=f"⏳ Pending")
        with col_status:
            if status == "failed":
                st.error(msg if msg else "Encountered failure")
            elif status == "running":
                st.info(msg if msg else "Processing...")
            elif status == "done":
                st.caption(f"Completed • {msg}")
            else:
                st.caption(msg if msg else "Waiting...")


def render_metadata_card(meta: Dict[str, Any]) -> None:
    """Display video metadata in neat metrics cards."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Duration", f"{meta.get('duration_seconds', 0):.1f}s")
    with col2:
        width = meta.get("video_stream", {}).get("width")
        height = meta.get("video_stream", {}).get("height")
        st.metric("Resolution", f"{width}x{height}" if width and height else "Unknown")
    with col3:
        st.metric("Frame Rate", f"{meta.get('frame_rate', 0):.1f} fps")
    with col4:
        st.metric("File Size", f"{meta.get('file_size_mb', 0):.1f} MB")
