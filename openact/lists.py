"""Word lists: inline answer sets and items, and telling a CSV path from inline text."""

from __future__ import annotations

# csvfmt holds the inline format because the Settings CSV's answers column uses it too, and csvfmt may import only
# the standard library.
from .csvfmt import parse_inline  # noqa: F401


def is_csv_path(text: str) -> bool:
    """True when the stripped text is a single line ending in ".csv", in any case. Anything else is inline text."""
    text = text.strip()
    return "\n" not in text and "\r" not in text and text.lower().endswith(".csv")
