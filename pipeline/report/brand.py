"""Groundtruth's mark, shared by the hosted site and every report.

A white "G" (the camera's lens) and an amber "T" read together as GT. The T's stem drops
onto a ground line, and the dot where the strokes meet is a worker placed on the ground
plane: the system's job, finding what is true on the ground, hidden in its initials.
"""

from urllib.parse import quote

NAME = "Groundtruth"
AMBER = "#F7B32B"
_PATHS = (
    '<rect width="64" height="64" rx="15" fill="#14171c"/>'
    '<path d="M44.5 19A17 17 0 1 0 48 31.5" fill="none" stroke="#eef1f5" stroke-width="5.5" '
    'stroke-linecap="round"/>'
    f'<path d="M31 31.5h19M41 31.5v16.5" fill="none" stroke="{AMBER}" stroke-width="5.5" '
    'stroke-linecap="round"/>'
    f'<path d="M28 51.5h26" stroke="{AMBER}" stroke-width="3" stroke-linecap="round"/>'
    '<circle cx="31" cy="31.5" r="3.6" fill="#eef1f5"/>'
)


def logo(size: int = 32, label: str = "") -> str:
    """The mark as inline SVG; decorative unless a label is given."""
    a11y = f'role="img" aria-label="{label}"' if label else 'aria-hidden="true"'
    return (
        f'<svg class="logo-mark" viewBox="0 0 64 64" width="{size}" height="{size}" {a11y}>'
        f"{_PATHS}</svg>"
    )


FAVICON = "data:image/svg+xml," + quote(
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">{_PATHS}</svg>'
)


def wordmark(size: int = 30) -> str:
    """Mark plus name: "ground" strong, "truth" light, read as one word."""
    return f'{logo(size)}<span class="wordmark"><b>ground</b><span>truth</span></span>'
