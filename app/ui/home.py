"""
app/ui/home.py — Home page for uploading video and setting editing requirements.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import List
import streamlit as st

from app.config import get_config
from app.providers.ollama_provider import OllamaProvider


def render() -> None:
    """Render the Home page UI."""
    st.title("🎬 AI Video Studio")
    st.markdown("Your local-first, AI-powered video editing studio.")

    cfg = get_config()
    provider = OllamaProvider(host=cfg.ollama_host, model=cfg.ollama_model)

    # Provider status banner in sidebar
    with st.sidebar:
        st.subheader("System Status")
        is_ollama_up = provider.is_available()
        if is_ollama_up:
            st.success(f"🟢 Ollama Online (`{cfg.ollama_model}`)")
            models = provider.get_available_models()
            if models:
                st.caption(f"Installed models: {', '.join(models)}")
        else:
            st.warning(f"🟡 Ollama not detected at `{cfg.ollama_host}`.\nEnsure Ollama is running.")

        st.markdown("---")
        st.markdown("**Device:** Apple Silicon (Unified Memory)")
        st.markdown(f"**Whisper:** `{cfg.whisper_model.split('/')[-1]}`")

    # Form inputs
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("1. Source Footage")
        uploaded_files = st.file_uploader(
            "Upload MP4 or MOV video(s)",
            type=["mp4", "mov"],
            accept_multiple_files=True,
            help="Upload primary footage (A-roll) and optional supplementary clips (B-roll, cutaways).",
        )

        existing_paths = st.session_state.get("source_video_paths")
        existing_single = st.session_state.get("source_video_path")

        if uploaded_files:
            if len(uploaded_files) == 1:
                st.video(uploaded_files[0])
                st.caption(f"📄 {uploaded_files[0].name} ({uploaded_files[0].size / (1024*1024):.1f} MB)")
            else:
                st.info(f"📹 {len(uploaded_files)} source videos selected")
                for idx, uf in enumerate(uploaded_files):
                    tag = "Primary (A-Roll)" if idx == 0 else f"Clip {idx+1}"
                    with st.expander(f"🎬 Source {idx} [{tag}]: {uf.name} ({uf.size / (1024*1024):.1f} MB)", expanded=(idx == 0)):
                        st.video(uf)
        elif existing_paths and isinstance(existing_paths, list):
            st.info(f"📁 {len(existing_paths)} source videos currently loaded")
            for idx, p in enumerate(existing_paths):
                if Path(p).exists():
                    with st.expander(f"🎬 Source {idx}: {Path(p).name}", expanded=(idx == 0)):
                        st.video(str(p))
        elif existing_single and Path(existing_single).exists():
            st.info(f"📁 Current video loaded: `{Path(existing_single).name}`")
            st.video(str(existing_single))

    with col2:
        st.subheader("2. Editing Instructions")
        default_prompt = st.session_state.get(
            "user_instruction",
            (
                "Create a high-energy dynamic short. Identify the strongest opening hook, "
                "remove long pauses, add animated captions with word highlighting, "
                "and apply subtle zooms on key statements."
            ),
        )
        prompt = st.text_area(
            "Natural Language Editing Prompt",
            value=default_prompt,
            height=120,
        )

        profiles = ["dynamic_shorts", "long_form", "cinematic"]
        saved_profile = st.session_state.get("profile", "dynamic_shorts")
        profile_idx = profiles.index(saved_profile) if saved_profile in profiles else 0

        profile = st.selectbox(
            "Editing Profile",
            options=profiles,
            index=profile_idx,
            format_func=lambda x: {
                "dynamic_shorts": "Dynamic Shorts (9:16 Vertical, 15-60s)",
                "long_form": "Long-form Explanatory (16:9, Natural Pacing)",
                "cinematic": "Cinematic Storytelling [Experimental]",
            }[x],
        )

        saved_dur = int(st.session_state.get("target_duration", 45))
        target_duration = st.slider(
            "Target Duration (seconds)",
            min_value=10,
            max_value=180,
            value=saved_dur,
            step=5,
        )

        col_a, col_b = st.columns(2)
        ratios = ["9:16", "16:9", "1:1"]
        saved_ar = st.session_state.get("aspect_ratio", "9:16")
        ar_idx = ratios.index(saved_ar) if saved_ar in ratios else (0 if profile == "dynamic_shorts" else 1)

        with col_a:
            aspect_ratio = st.selectbox(
                "Aspect Ratio",
                options=ratios,
                index=ar_idx,
            )

        styles = ["dynamic", "clean", "minimal"]
        saved_style = st.session_state.get("caption_style", "dynamic")
        style_idx = styles.index(saved_style) if saved_style in styles else 0

        with col_b:
            caption_style = st.selectbox(
                "Caption Style",
                options=styles,
                index=style_idx,
            )

        enable_music = st.checkbox(
            "Enable Background Music (if available in assets)",
            value=st.session_state.get("enable_music", False),
        )

    st.markdown("---")
    start_btn = st.button("🚀 Start AI Editing Workflow", type="primary", use_container_width=True)

    if start_btn:
        videos_to_use: List[str] = []
        temp_dir = Path(tempfile.gettempdir()) / "ai_studio_uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)

        if uploaded_files:
            for idx, uf in enumerate(uploaded_files):
                temp_video_path = temp_dir / f"upload_{idx}_{uuid.uuid4().hex[:6]}_{uf.name}"
                with open(temp_video_path, "wb") as f:
                    f.write(uf.getbuffer())
                videos_to_use.append(str(temp_video_path))
        elif existing_paths and isinstance(existing_paths, list):
            videos_to_use = [str(p) for p in existing_paths if Path(p).exists()]
        elif existing_single and Path(existing_single).exists():
            videos_to_use = [str(existing_single)]
        else:
            st.error("Please upload at least one video file to begin.")
            return

        job_id = str(uuid.uuid4())
        st.session_state["job_id"] = job_id
        st.session_state["source_video_paths"] = videos_to_use
        st.session_state["source_video_path"] = videos_to_use[0]
        st.session_state["user_instruction"] = prompt
        st.session_state["profile"] = profile
        st.session_state["target_duration"] = float(target_duration)
        st.session_state["aspect_ratio"] = aspect_ratio
        st.session_state["caption_style"] = caption_style
        st.session_state["enable_music"] = enable_music
        st.session_state["page"] = "processing"
        st.rerun()
