import re
import time
from typing import List, Tuple
from prompt_toolkit.formatted_text import AnyFormattedText, to_formatted_text
from prompt_toolkit.shortcuts import print_formatted_text
from vocode.ui.terminal.colors import get_console_style

from vocode.ui.terminal.colors import render_markdown


def _count_trailing_newlines(s: str) -> int:
    count = 0
    for char in reversed(s):
        if char == '\n':
            count += 1
        else:
            break
    return count


def _strip_trailing_newlines_from_formatted(formatted: List[Tuple[str, str]]):
    while formatted:
        style, text = formatted[-1]
        stripped_text = text.rstrip('\n')
        
        if stripped_text:
            formatted[-1] = (style, stripped_text)
            break
        else:
            formatted.pop()


class MessageBuffer:
    def __init__(self, speaker: str) -> None:
        self.speaker = speaker
        self._prefix = f"{self.speaker}: "
        self.full_text = ""

    def append(self, chunk: str) -> AnyFormattedText:
        if not chunk:
            return []

        num_newlines_in_chunk = chunk.count("\n")

        # Accumulate appended chunks
        self.full_text += chunk

        # Format whole property
        full_formatted = to_formatted_text(render_markdown(self.full_text, prefix=self._prefix))
        mutable_formatted = list(full_formatted)

        # Adjust trailing newlines to match the source chunk.
        num_trailing_newlines_in_chunk = _count_trailing_newlines(chunk)
        _strip_trailing_newlines_from_formatted(mutable_formatted)
        if num_trailing_newlines_in_chunk > 0:
            if mutable_formatted:
                style, text = mutable_formatted[-1]
                mutable_formatted[-1] = (style, text + '\n' * num_trailing_newlines_in_chunk)
            else:
                mutable_formatted.append(("", '\n' * num_trailing_newlines_in_chunk))

        # Two-step process: find the start of the lines to return, then slice.
        newlines_to_skip = num_newlines_in_chunk + 1
        
        start_fragment_idx = 0
        start_char_idx_in_fragment = 0

        # Find the starting point by iterating backwards.
        newlines_skipped = 0
        found_start = False
        for i in range(len(mutable_formatted) - 1, -1, -1):
            _, text_part = mutable_formatted[i]

            pos = len(text_part)
            while True:
                last_newline_pos = text_part.rfind('\n', 0, pos)
                if last_newline_pos == -1:
                    break  # No more newlines in this fragment

                newlines_skipped += 1
                pos = last_newline_pos

                if newlines_skipped >= newlines_to_skip:
                    start_fragment_idx = i
                    start_char_idx_in_fragment = pos + 1
                    found_start = True
                    break

            if found_start:
                break

        # Generate the new formatted sequence from the starting point.
        fragments_for_return: List[Tuple[str, str]] = []
        if start_fragment_idx < len(mutable_formatted):
            # Add the partial starting fragment.
            style, text = mutable_formatted[start_fragment_idx]
            substring = text[start_char_idx_in_fragment:]
            if substring:
                fragments_for_return.append((style, substring))
            
            # Add all subsequent fragments.
            fragments_for_return.extend(mutable_formatted[start_fragment_idx + 1:])
            
        return fragments_for_return
