"""
app/main.py — Main entry point for the Streamlit AI Video Studio application.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from app.config import get_config
from app.ui import export, home, preview, processing
from app.utils.db import init_db
from app.utils.logger import get_logger

logger = get_logger("ai_video_studio")


def main() -> None:
    """Initialize and route the Streamlit multi-page interface."""
    st.set_page_config(
        page_title="AI Video Studio",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize SQLite database
    cfg = get_config()
    init_db(cfg.db_path)

    # State routing setup
    if "page" not in st.session_state:
        st.session_state["page"] = "home"
    if "job_id" not in st.session_state:
        st.session_state["job_id"] = None

    # Sidebar persistent navigation
    with st.sidebar:
        st.title("🎬 Studio Hub")
        if st.session_state.get("pipeline_results"):
            st.success("Deliverables Ready!")
            if st.button("📦 Go to Downloads", type="primary", use_container_width=True):
                st.session_state["page"] = "export"
                st.rerun()
            if st.button("👁️ Go to Preview", use_container_width=True):
                st.session_state["page"] = "preview"
                st.rerun()
        if st.button("🏠 Home (New Video)", use_container_width=True):
            st.session_state["page"] = "home"
            st.rerun()
        st.markdown("---")
        st.caption("AI Video Studio • 100% Local Apple Silicon")

    current_page = st.session_state.get("page", "home")

    if current_page == "home":
        home.render()
    elif current_page == "processing":
        processing.render()
    elif current_page == "preview":
        preview.render()
    elif current_page == "export":
        export.render()
    else:
        st.error(f"Unknown page: {current_page}")
        if st.button("Reset to Home"):
            st.session_state["page"] = "home"
            st.rerun()


if __name__ == "__main__":
    main()
