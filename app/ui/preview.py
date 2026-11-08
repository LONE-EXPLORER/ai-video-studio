"""
app/ui/preview.py — Preview generated edit, review plan, inspect quality checks.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ui.components import render_metadata_card


def render() -> None:
    """Render the Preview page UI."""
    results = st.session_state.get("pipeline_results")
    if not results:
        st.warning("No generated results to preview.")
        if st.button("Return to Home"):
            st.session_state["page"] = "home"
            st.rerun()
        return

    st.title("👁️ Review & Approve Generated Edit")

    col1, col2 = st.columns([6, 4], gap="large")

    with col1:
        st.subheader("Video Preview")
        preview_path = results.get("preview_video")
        if preview_path and preview_path.exists():
            st.video(str(preview_path))
        else:
            st.info("Preview video file not found.")

        # Direct Download Buttons right here on Preview page
        final_video = results.get("final_video")
        fcpxml_path = results.get("fcpxml_path")

        st.markdown("#### ⬇️ Instant Downloads")
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            if final_video and final_video.exists():
                with open(final_video, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Final MP4 Reel",
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
                        label="⬇️ Download DaVinci FCPXML",
                        data=f.read(),
                        file_name="timeline.fcpxml",
                        mime="application/xml",
                        use_container_width=True,
                    )

        # Quality Report
        quality_report = results.get("quality_report")
        if quality_report:
            st.markdown("### Quality Verification")
            for chk in quality_report.checks:
                status_icon = "✅" if chk.passed else "⚠️"
                st.markdown(f"- **{status_icon} {chk.name}:** {chk.detail}")

    with col2:
        st.subheader("AI Editorial Plan")
        plan = results.get("plan")
        if plan:
            st.markdown(f"**Hook Summary:** {plan.hook_summary}")
            st.markdown(f"**Rationale:** {plan.edit_rationale}")
            st.markdown(f"**Target Duration:** {plan.target_duration_seconds}s | **Actual Duration:** {plan.total_duration:.1f}s")

            st.markdown("#### Selected Clips")
            clip_data = [
                {
                    "Order": c.order,
                    "Start": f"{c.source_start:.1f}s",
                    "End": f"{c.source_end:.1f}s",
                    "Duration": f"{c.duration:.1f}s",
                    "Reason": c.reason,
                    "Zoom": "Yes" if c.zoom.enabled else "No",
                }
                for c in plan.clips
            ]
            st.dataframe(pd.DataFrame(clip_data), hide_index=True, use_container_width=True)

        with st.expander("Full Source Transcript"):
            transcript = results.get("transcript")
            if transcript:
                st.text(transcript.full_text())

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬅️ Adjust Settings & Re-edit", use_container_width=True):
            st.session_state["page"] = "home"
            st.rerun()
    with c2:
        if st.button("✅ Approve & Go to Exports", type="primary", use_container_width=True):
            st.session_state["page"] = "export"
            st.rerun()
