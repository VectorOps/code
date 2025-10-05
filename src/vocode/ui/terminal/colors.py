from __future__ import annotations

from typing import List, Optional, Tuple, Any
from enum import Enum
import json
import re
from pygments.lexers.markup import MarkdownLexer

from prompt_toolkit.formatted_text import (
    AnyFormattedText,
    HTML,
    PygmentsTokens,
    merge_formatted_text,
)
from prompt_toolkit.styles import Style, merge_styles
from prompt_toolkit.styles.pygments import style_from_pygments_cls

import pygments
from pygments.lexers import get_lexer_by_name
from pygments.lexers.special import TextLexer
from pygments.styles import get_style_by_name


# Precompiled regexes for markdown parsing
_FENCE_OPEN_RE = re.compile(r"^\s*```([A-Za-z0-9_\-]+)?\s*$")
_FENCE_CLOSE_RE = re.compile(r"^\s*```\s*$")

_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
# Basic italic (avoid eating bold): match single-asterisk groups not surrounded by another '*'
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")

# Map fenced code languages to pygments lexer names
_CODE_FENCE_LANG_MAP = {
    # v4a uses ```patch ...```; highlight with diff lexer
    "patch": "diff",
}


class MarkdownBlockType(str, Enum):
    TEXT = "text"
    CODE = "code"


def _md_inline_to_html(text: str) -> PygmentsTokens:
    """
    Tokenize a subset of inline markdown using pygments' MarkdownLexer and
    return PygmentsTokens for prompt_toolkit rendering.
    Handles emphasis, strong, and inline code spans.
    """
    lexer = MarkdownLexer()
    tokens = list(pygments.lex(text, lexer))
    return PygmentsTokens(tokens)


def _split_markdown_blocks(
    text: str,
) -> List[Tuple[MarkdownBlockType, Optional[str], str]]:
    """
    Split markdown into blocks: list of tuples (kind, lang, content)
    kind is a MarkdownBlockType, lang is None for non-code.
    """
    lines = text.splitlines(keepends=True)
    blocks: List[Tuple[MarkdownBlockType, Optional[str], str]] = []
    in_code = False
    code_lang: Optional[str] = None
    buf: List[str] = []

    def flush_text():
        nonlocal buf
        if buf:
            blocks.append((MarkdownBlockType.TEXT, None, "".join(buf)))
            buf = []

    def flush_code():
        nonlocal buf, code_lang
        if buf:
            blocks.append((MarkdownBlockType.CODE, code_lang, "".join(buf)))
            buf = []
        code_lang = None

    for ln in lines:
        if not in_code:
            buf.append(ln)

            m = _FENCE_OPEN_RE.match(ln.strip("\r\n"))
            if m:
                flush_text()
                in_code = True
                code_lang = (m.group(1) or "").strip() or None
        else:
            if _FENCE_CLOSE_RE.match(ln.strip("\r\n")):
                flush_code()
                in_code = False

            buf.append(ln)

    # Flush tail
    if in_code:
        flush_code()
    else:
        flush_text()

    return blocks


def render_markdown(text: str, prefix: Optional[str] = None) -> AnyFormattedText:
    """
    Render markdown into prompt_toolkit formatted fragments without using HTML.
    - Code fences ```lang ... ``` highlighted via Pygments (PygmentsTokens).
    - Inline markdown (bold/italic/`code`) via Pygments MarkdownLexer.
    If prefix is provided, it is prepended as plain text at the beginning.
    """
    blocks = _split_markdown_blocks(text or "")
    parts: List[AnyFormattedText] = []
    prefixed = False

    for kind, lang, content in blocks:
        if kind == MarkdownBlockType.TEXT:
            s = content
            if prefix and not prefixed:
                s = f"{prefix}{s}"
                prefixed = True
            parts.append(_md_inline_to_html(s))
        else:
            # code block
            if prefix and not prefixed:
                parts.append(prefix)
                prefixed = True
            lexer = None
            if lang:
                # Normalize language and resolve aliases before resolving lexer
                normalized_lang = (lang or "").strip().lower()
                normalized_lang = _CODE_FENCE_LANG_MAP.get(normalized_lang, normalized_lang)
                try:
                    lexer = get_lexer_by_name(normalized_lang, stripall=False)
                except Exception:
                    lexer = None
            if lexer is None:
                lexer = TextLexer(stripall=False)
            try:
                tokens = list(pygments.lex(content, lexer))
            except Exception:
                # Fallback: treat as plain text if lexing fails
                parts.append(content)
            else:
                parts.append(PygmentsTokens(tokens))

    if not parts:
        # Empty content, still show prefix if provided
        if prefix:
            return prefix
        return ""

    return merge_formatted_text(parts)


def s_endswith_newline(parts: List[AnyFormattedText]) -> bool:
    """
    Helper to detect if last HTML fragment ends with a newline.
    """
    if not parts:
        return False
    frag = parts[-1]
    # Best-effort: HTML(text) stores the original string; if not available, ignore.
    try:
        if isinstance(frag, HTML):
            # access original value through .value or ._value depending on version
            v = getattr(frag, "value", None) or getattr(frag, "_value", "")
            return str(v).endswith("\n")
    except Exception:
        pass
    return False


def get_last_non_empty_line(
    lines: List[List[Tuple[str, str]]],
) -> Optional[List[Tuple[str, str]]]:
    """
    Given a list of formatted text lines, return the last line that is not empty.
    A line is considered empty if it contains only whitespace.
    """
    for line in reversed(lines):
        # Join all text fragments in the line
        line_text = "".join(text for _, text in line)
        if line_text.strip():
            # Found a non-empty line
            return line
    return None


# Style for console output: merge base style with a pygments theme.
# You can change "default" to a different pygments style (e.g., "monokai", "friendly", etc.)
_PYGMENTS_STYLE = style_from_pygments_cls(get_style_by_name("default"))
_BASE_STYLE = Style(
    [
        # You can extend non-pygments styles here if needed
        # ("class:speaker.user", "bold"),
        # ("class:speaker.agent", "bold"),
    ]
)
_CONSOLE_STYLE = merge_styles([_BASE_STYLE, _PYGMENTS_STYLE])


def get_console_style() -> Style:
    """
    Return a prompt_toolkit Style to be used with print_formatted_text when rendering markdown.
    """
    return _CONSOLE_STYLE


def render_json(data: Any) -> AnyFormattedText:
    """
    Pretty-print JSON with syntax highlighting.
    - If data is str/bytes, attempt json.loads; on failure, render as plain text.
    - Otherwise, json.dumps with indent=2 (ensure_ascii=False).
    Falls back to plain text tokens if serialization or lexing fails.
    """
    text: str
    try:
        if isinstance(data, (bytes, bytearray)):
            s = data.decode("utf-8", errors="replace")
            try:
                obj = json.loads(s)
                text = json.dumps(obj, indent=2, ensure_ascii=False)
            except Exception:
                tokens = list(pygments.lex(s, TextLexer()))
                return PygmentsTokens(tokens)
        elif isinstance(data, str):
            try:
                obj = json.loads(data)
                text = json.dumps(obj, indent=2, ensure_ascii=False)
            except Exception:
                tokens = list(pygments.lex(data, TextLexer()))
                return PygmentsTokens(tokens)
        else:
            # Best-effort pretty print; may raise if not JSON-serializable
            text = json.dumps(data, indent=2, ensure_ascii=False)
    except Exception:
        tokens = list(pygments.lex(str(data), TextLexer()))
        return PygmentsTokens(tokens)

    try:
        lexer = get_lexer_by_name("json")
    except Exception:
        lexer = TextLexer()
    tokens = list(pygments.lex(text, lexer))
    return PygmentsTokens(tokens)
