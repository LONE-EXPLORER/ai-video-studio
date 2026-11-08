"""
app/ui/export.py — Export page for downloading final assets and reports.
"""

from __future__ import annotations

import streamlit as st


def render() -> None:
    """Render the Export page UI."""
    results = st.session_state.get("pipeline_results")
    if not results:
        st.warning("No export data available.")
        if st.button("Return to Home"):
            st.session_state["page"] = "home"
            st.rerun()
        return

    st.title("📦 Export Video & Studio Assets")
    st.markdown("Your video has been rendered and verified. Download your deliverables below:")

    final_video = results.get("final_video")
    srt_path = results.get("srt_path")
    ass_path = results.get("ass_path")
    plan_path = results.get("plan_path")
    quality_path = results.get("quality_report_path")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("🎥 Final Media")
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
            st.caption(f"Location: `{final_video}`")

    with col2:
        st.subheader("📄 Subtitles & Project Metadata")
        if srt_path and srt_path.exists():
            with open(srt_path, "r", encoding="utf-8") as f:
                st.download_button(
                    label="⬇️ Download SRT Subtitles",
                    data=f.read(),
                    file_name="subtitles.srt",
                    mime="text/plain",
                    use_container_width=True,
                )

        if ass_path and ass_path.exists():
            with open(ass_path, "r", encoding="utf-8") as f:
                st.download_button(
                    label="⬇️ Download ASS Formatted Subtitles",
                    data=f.read(),
                    file_name="subtitles.ass",
                    mime="text/plain",
                    use_container_width=True,
                )

        fcpxml_path = results.get("fcpxml_path")
        if fcpxml_path and fcpxml_path.exists():
            with open(fcpxml_path, "r", encoding="utf-8") as f:
                st.download_button(
                    label="⬇️ Download DaVinci Resolve Timeline (.fcpxml)",
                    data=f.read(),
                    file_name="timeline.fcpxml",
                    mime="application/xml",
                    use_container_width=True,
                )

        if plan_path and plan_path.exists():
            with open(plan_path, "r", encoding="utf-8") as f:
                st.download_button(
                    label="⬇️ Download Editing Plan (JSON)",
                    data=f.read(),
                    file_name="editing_plan.json",
                    mime="application/json",
                    use_container_width=True,
                )

    st.markdown("---")
    if st.button("🎬 Start New Project", use_container_width=True):
        st.session_state.clear()
        st.session_state["page"] = "home"
        st.rerun()
