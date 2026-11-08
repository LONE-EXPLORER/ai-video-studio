"""
app/ui/processing.py — Live workflow execution and progress display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import time
import streamlit as st

from app.config import get_config
from app.pipeline.orchestrator import PipelineOrchestrator
from app.providers.ollama_provider import OllamaProvider
from app.ui.components import render_stage_progress


def render() -> None:
    """Render the Processing page UI."""
    job_id = st.session_state.get("job_id")
    if not job_id:
        st.warning("No active job found.")
        if st.button("Return to Home"):
            st.session_state["page"] = "home"
            st.rerun()
        return

    st.title("⚙️ AI Studio Pipeline")
    st.caption(f"Processing Job ID: `{job_id}`")

    stages_tracker: List[Dict[str, Any]] = [
        {"name": "ingest", "status": "pending", "message": "Pending validation"},
        {"name": "transcribe", "status": "pending", "message": "Pending audio extraction"},
        {"name": "analyze", "status": "pending", "message": "Pending frame analysis"},
        {"name": "plan", "status": "pending", "message": "Pending editorial intelligence"},
        {"name": "edit", "status": "pending", "message": "Pending clip cutting & audio processing"},
        {"name": "render", "status": "pending", "message": "Pending final render"},
        {"name": "quality", "status": "pending", "message": "Pending quality inspection"},
        {"name": "preview", "status": "pending", "message": "Pending web preview generation"},
    ]

    if "job_start_time" not in st.session_state or st.session_state.get("active_job_id") != job_id:
        st.session_state["job_start_time"] = time.time()
        st.session_state["active_job_id"] = job_id

    job_start_time = st.session_state["job_start_time"]
    cfg = get_config()
    provider = OllamaProvider(host=cfg.ollama_host, model=cfg.ollama_model, fallback_model=cfg.ollama_fallback_model)
    orchestrator = PipelineOrchestrator(provider=provider)

    status_placeholder = st.empty()
    log_placeholder = st.empty()

    def _render_live_log(expanded: bool = False) -> None:
        log_file = Path(cfg.jobs_dir) / job_id / "pipeline.log"
        if log_file.exists():
            with log_placeholder.container():
                with st.expander("📋 Live Technical Execution Log", expanded=expanded):
                    st.code(log_file.read_text(encoding="utf-8"))

    def update_stage(stage_name: str, status: str, message: str) -> None:
        for s in stages_tracker:
            if s["name"] == stage_name:
                s["status"] = status
                s["message"] = message
                break
        with status_placeholder.container():
            render_stage_progress(stages_tracker, start_time=job_start_time)
        _render_live_log(expanded=False)

    with status_placeholder.container():
        render_stage_progress(stages_tracker, start_time=job_start_time)
    _render_live_log(expanded=False)

    with st.spinner("Processing video through AI Video Studio..."):
        try:
            source_paths = st.session_state.get("source_video_paths")
            if not source_paths:
                source_paths = [st.session_state["source_video_path"]]
            source_paths = [Path(p) for p in source_paths]

            results = orchestrator.run_job(
                job_id=job_id,
                source_video_path=source_paths[0],
                source_video_paths=source_paths,
                user_instruction=st.session_state["user_instruction"],
                profile=st.session_state["profile"],
                target_duration=st.session_state["target_duration"],
                aspect_ratio=st.session_state["aspect_ratio"],
                caption_style=st.session_state["caption_style"],
                enable_music=st.session_state["enable_music"],
                progress_callback=update_stage,
            )
            st.session_state["pipeline_results"] = results
            st.success("🎉 Workflow completed successfully! Download your deliverables below or preview the edit:")
            _render_live_log(expanded=False)

            # Direct Download & Navigation options
            final_video = results.get("final_video")
            fcpxml_path = results.get("fcpxml_path")

            st.markdown("### ⬇️ Quick Downloads")
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                if final_video and final_video.exists():
                    with open(final_video, "rb") as f:
                        st.download_button(
                            label="⬇️ Download Final MP4 Video",
                            data=f.read(),
                            file_name=final_video.name,
                            mime="video/mp4",
                            type="primary",
                            use_container_width=True,
                        )
            with dcol2:
                if fcpxml_path and fcpxml_path.exists():
                    with open(fcpxml_path, "r", encoding="utf-8") as f:
                        st.download_button(
                            label="⬇️ Download DaVinci Timeline (.fcpxml)",
                            data=f.read(),
                            file_name="timeline.fcpxml",
                            mime="application/xml",
                            use_container_width=True,
                        )

            st.markdown("---")
            nav_col1, nav_col2 = st.columns(2)
            with nav_col1:
                if st.button("👉 View Video Preview & Plan", type="secondary", use_container_width=True):
                    st.session_state["page"] = "preview"
                    st.rerun()
            with nav_col2:
                if st.button("📦 Full Export Hub (SRT, ASS, Plan JSON)", type="primary", use_container_width=True):
                    st.session_state["page"] = "export"
                    st.rerun()

        except Exception as exc:
            st.error(f"Pipeline execution encountered an error: {exc}")
            _render_live_log(expanded=True)

            col_err1, col_err2 = st.columns(2)
            with col_err1:
                if st.button("🔄 Retry This Job", type="primary", use_container_width=True):
                    st.rerun()
            with col_err2:
                if st.button("⬅️ Back to Home (Keep Settings)", use_container_width=True):
                    st.session_state["page"] = "home"
                    st.rerun()
