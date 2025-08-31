from __future__ import annotations

from typing import List, Optional, Tuple
from enum import Enum
import re
from html import escape as html_escape

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


class MarkdownBlockType(str, Enum):
    TEXT = "text"
    CODE = "code"


def _md_inline_to_html(text: str) -> str:
    """
    Convert a subset of inline markdown to prompt_toolkit HTML.
    - **bold** => <b>...</b>
    - *italic* => <i>...</i>
    - `code`   => styled code span
    """
    # Escape first to avoid HTML injection
    s = html_escape(text)

    # Inline code: soft-styled using fg/bg (avoid 'ansi*' tags as requested)
    def repl_code(m: re.Match) -> str:
        code = m.group(1)
        return f'<style fg="#00dd88" bg="#1e1e1e">{html_escape(code)}</style>'

    s = _INLINE_CODE_RE.sub(repl_code, s)
    s = _BOLD_RE.sub(r"<b>\1</b>", s)
    s = _ITALIC_RE.sub(r"<i>\1</i>", s)
    return s


def _split_markdown_blocks(text: str) -> List[Tuple[MarkdownBlockType, Optional[str], str]]:
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
    Render a markdown string into prompt_toolkit formatted fragments.
    - Code fences ```lang ... ``` highlighted via Pygments and wrapped as PygmentsTokens.
    - Inline markdown (bold/italic/`code`) mapped to HTML tags/styles.
    - No ANSI escapes are produced.
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
            html = _md_inline_to_html(s)
            parts.append(HTML(html))
        else:
            # code block
            if prefix and not prefixed:
                parts.append(HTML(html_escape(prefix)))
                prefixed = True
            lexer = None
            if lang:
                try:
                    lexer = get_lexer_by_name(lang, stripall=False)
                except Exception:
                    lexer = None
            if lexer is None:
                lexer = TextLexer(stripall=False)
            try:
                tokens = list(pygments.lex(content, lexer))
            except Exception:
                # Fallback: treat as plain text if lexing fails
                parts.append(HTML(html_escape(content)))
            else:
                parts.append(PygmentsTokens(tokens))

    if not parts:
        # Empty content, still show prefix if provided
        if prefix:
            return HTML(html_escape(prefix))
        return HTML("")

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
