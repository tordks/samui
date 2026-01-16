"""Segmentation mode toggle component."""

import streamlit as st

from samui_frontend.models import SegmentationMode


def render_mode_toggle(
    key: str = "mode_radio",
    include_point: bool = False,
) -> SegmentationMode:
    """Render segmentation mode toggle and return selected mode.

    Args:
        key: Unique key for the radio button widget.
        include_point: If True, include POINT mode in the options.

    Returns:
        The currently selected SegmentationMode.
    """
    mode_options = {
        SegmentationMode.INSIDE_BOX: "Inside Box",
        SegmentationMode.FIND_ALL: "Find All",
    }
    mode_descriptions = {
        SegmentationMode.INSIDE_BOX: "Segment objects inside each bounding box",
        SegmentationMode.FIND_ALL: "Find all instances matching text prompt and/or exemplars",
    }

    if include_point:
        mode_options[SegmentationMode.POINT] = "Point"
        mode_descriptions[SegmentationMode.POINT] = "Segment using positive and negative point prompts"

    # Map enum to index
    mode_list = list(mode_options.keys())
    current_mode = st.session_state.segmentation_mode
    current_index = mode_list.index(current_mode) if current_mode in mode_list else 0

    selected_label = st.radio(
        "Segmentation Mode",
        options=list(mode_options.values()),
        index=current_index,
        horizontal=True,
        key=key,
    )

    # Map label back to enum
    label_to_mode = {v: k for k, v in mode_options.items()}
    mode = label_to_mode[selected_label]

    # Update session state if mode changed
    if mode != st.session_state.segmentation_mode:
        st.session_state.segmentation_mode = mode
        st.rerun()

    st.caption(mode_descriptions[mode])

    return mode
