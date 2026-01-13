"""Arrow navigator component for cycling through items."""

from typing import Any

import streamlit as st


def arrow_navigator(items: list[Any], key_prefix: str) -> int | None:
    """Display arrow navigation for cycling through a list of items.

    Renders: < [current/total] > with left/right buttons.
    Stores current index in session state using key_prefix.

    Args:
        items: List of items to navigate through.
        key_prefix: Unique prefix for session state keys.

    Returns:
        Current index into items, or None if items is empty.
    """
    if not items:
        st.info("No items to navigate.")
        return None

    state_key = f"{key_prefix}_index"

    # Initialize session state if needed
    if state_key not in st.session_state:
        st.session_state[state_key] = 0

    current_index = st.session_state[state_key]

    # Clamp index to valid range (handles case where items list shrinks)
    if current_index >= len(items):
        current_index = len(items) - 1
        st.session_state[state_key] = current_index

    total = len(items)

    # Single item: just show count, no buttons
    if total == 1:
        st.write("**1 / 1**")
        return 0

    # Multiple items: show navigation buttons
    col_left, col_count, col_right = st.columns([1, 2, 1])

    with col_left:
        if st.button("◀", key=f"{key_prefix}_prev", disabled=(current_index == 0)):
            st.session_state[state_key] = current_index - 1
            st.rerun()

    with col_count:
        st.write(f"**{current_index + 1} / {total}**")

    with col_right:
        if st.button("▶", key=f"{key_prefix}_next", disabled=(current_index == total - 1)):
            st.session_state[state_key] = current_index + 1
            st.rerun()

    return st.session_state[state_key]
