"""Shared constants for the frontend."""

# Color palette for bounding boxes (cycle through these)
BBOX_COLORS = [
    "#ff4b4b",  # red
    "#4bff4b",  # green
    "#4b4bff",  # blue
    "#ffff4b",  # yellow
    "#ff4bff",  # magenta
    "#4bffff",  # cyan
]

# Semantic colors for annotation types
COLOR_POSITIVE_EXEMPLAR = "#4bff4b"  # green
COLOR_NEGATIVE_EXEMPLAR = "#ff4b4b"  # red

# API timeouts
API_TIMEOUT_READ = 10.0
API_TIMEOUT_WRITE = 30.0
