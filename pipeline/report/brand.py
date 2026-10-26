"""Ground Truth's mark, shared by the hosted site and every report.

A GT monogram in flat, mitred strokes on a charcoal tile. The off-white G is the camera's
view of the site; the amber T is its crossbar and a plumb stem dropping onto the amber
ground line, cutting a gap through the G's base on the way. The off-white dot standing on
the ground line is a worker placed on the ground plane: the system's job, finding what is
true on the ground, inside its initials.
"""

from urllib.parse import quote

NAME = "Ground Truth"
TILE, WHITE, AMBER = "#14171C", "#EEF1F5", "#F7B32B"
_PATHS = (
    f'<rect width="64" height="64" rx="14" fill="{TILE}"/>'
    '<g fill="none" stroke-width="5.5" stroke-linecap="butt" stroke-linejoin="miter">'
    f'<path d="M45 10H20l-6 6v18l6 6h15.5" stroke="{WHITE}"/>'
    f'<path d="M29.5 28.5h21M40 28.5v22" stroke="{AMBER}"/>'
    f'<path d="M11 52h42" stroke="{AMBER}" stroke-width="3"/></g>'
    f'<circle cx="21.5" cy="47.6" r="2.8" fill="{WHITE}"/>'
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
    """Mark plus the name in bold Space Grotesk capitals."""
    return f'{logo(size)}<span class="wordmark">GROUND TRUTH</span>'
