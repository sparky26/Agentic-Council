from __future__ import annotations

import re
from typing import Optional


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    """
    Collapse all runs of whitespace (spaces, tabs, newlines) into single spaces
    and strip leading/trailing spaces.

    Useful for cleaning LLM outputs before logging, evaluation, or UI display.
    """
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip()


def truncate_chars(
    text: str,
    max_chars: int,
    *,
    suffix: str = "…",
) -> str:
    """
    Truncate a string to at most `max_chars` characters.

    - If the text is shorter than max_chars, return it unchanged.
    - If truncation happens, append `suffix` (default: ellipsis).
    """
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if len(suffix) >= max_chars:
        # Degenerate case: suffix longer than limit
        return suffix[:max_chars]
    return text[: max_chars - len(suffix)] + suffix


_MARKDOWN_SPECIAL_CHARS = r"\`*_{}[]()#+-.!"


def safe_markdown(text: str) -> str:
    """
    Make a string safer for embedding in Markdown by escaping characters that
    often break formatting.

    This is deliberately simple and conservative; it's not a full Markdown
    sanitizer, just enough to avoid accidental bullet lists / headings when
    you don't want them.
    """
    if not text:
        return ""
    escaped = []
    for ch in text:
        if ch in _MARKDOWN_SPECIAL_CHARS:
            escaped.append("\\" + ch)
        else:
            escaped.append(ch)
    return "".join(escaped)


def strip_markdown(text: str) -> str:
    """
    Very rough Markdown "stripping" to get plain-ish text.

    - Removes leading '#' / '>' and bullet markers
    - Collapses multiple spaces
    - Does NOT attempt full Markdown parsing.
    """
    if not text:
        return ""

    # Remove common Markdown prefix characters at the start of lines
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.lstrip()
        for prefix in ("# ", "## ", "### ", "> ", "- ", "* ", "+ "):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :]
                break
        cleaned_lines.append(stripped)

    joined = "\n".join(cleaned_lines)
    return normalize_whitespace(joined)