import re
import time
from typing import List, Tuple
from prompt_toolkit.formatted_text import AnyFormattedText, to_formatted_text
from prompt_toolkit.shortcuts import print_formatted_text
from prompt_toolkit.formatted_text.utils import split_lines
from vocode.ui.terminal.colors import get_console_style

from vocode.ui.terminal.colors import render_markdown


class MessageBuffer:
    def __init__(self, speaker: str) -> None:
        self.speaker = speaker
        # Remove speaker/role prefix from rendered output
        self._prefix = ""
        self.full_text = ""
        self._cached_lines: List[List[Tuple[str, str]]] = []

    def append(
        self, chunk: str
    ) -> Tuple[List[List[Tuple[str, str]]], List[List[Tuple[str, str]]]]:
        if not chunk:
            return [], []

        self.full_text += chunk

        full_formatted = to_formatted_text(
            render_markdown(self.full_text, prefix=self._prefix)
        )
        new_lines = list(split_lines(full_formatted))

        # Determine start of changed lines
        first_diff_idx = 0
        limit = min(len(new_lines), len(self._cached_lines))
        while (
            first_diff_idx < limit
            and new_lines[first_diff_idx] == self._cached_lines[first_diff_idx]
        ):
            first_diff_idx += 1

        old_changed_lines = self._cached_lines[first_diff_idx:]
        new_changed_lines = new_lines[first_diff_idx:]
        self._cached_lines = new_lines

        return new_changed_lines, old_changed_lines
