"""Segmentation mode toggle component."""

import streamlit as st

from samui_frontend.models import SegmentationMode


def render_mode_toggle(key: str = "mode_radio") -> SegmentationMode:
    """Render segmentation mode toggle and return selected mode.

    Args:
        key: Unique key for the radio button widget.

    Returns:
        The currently selected SegmentationMode.
    """
    mode_options = {
        SegmentationMode.INSIDE_BOX: "Inside Box",
        SegmentationMode.FIND_ALL: "Find All",
    }
    mode_descriptions = {
        SegmentationMode.INSIDE_BOX: "Segment objects inside each bounding box",
        SegmentationMode.FIND_ALL: "Find all instances matching text prompt or exemplars",
    }

    col1, col2 = st.columns([1, 2])
    with col1:
        selected_label = st.radio(
            "Segmentation Mode",
            options=list(mode_options.values()),
            index=0 if st.session_state.segmentation_mode == SegmentationMode.INSIDE_BOX else 1,
            horizontal=True,
            key=key,
        )

    # Map label back to enum
    mode = SegmentationMode.INSIDE_BOX if selected_label == "Inside Box" else SegmentationMode.FIND_ALL

    # Update session state if mode changed
    if mode != st.session_state.segmentation_mode:
        st.session_state.segmentation_mode = mode
        st.rerun()

    with col2:
        st.caption(mode_descriptions[mode])

    return mode
