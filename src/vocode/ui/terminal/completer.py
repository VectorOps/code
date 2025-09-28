from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Union
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

from .commands import Commands, CommandCompletionProvider
from vocode.ui.base import UIState

# Provider signature: given the document, parsed args (excluding the command),
# and the word currently being completed (arg_prefix), return strings or Completion objects.
CommandCompletionProvider = Callable[
    [Document, List[str], str], Iterable[Union[str, Completion]]
]

# Optional non-command provider: given the document and current word prefix.
GeneralCompletionProvider = Callable[
    [Document, str], Iterable[Union[str, Completion]]
]


class TerminalCompleter(Completer):
    """
    Generic, nested completer for the terminal UI:
      - If the input starts with '/', offer command name completion.
      - If a command is present, delegate to a per-command provider if available,
        otherwise use a dummy provider (returns no suggestions).
      - If the input does not start with '/', optionally delegate to a general provider.
    """

    def __init__(
        self,
        ui: UIState,
        commands: Commands,
        default_command_provider: Optional[CommandCompletionProvider] = None,
        general_provider: Optional[GeneralCompletionProvider] = None,
    ) -> None:
        self._ui = ui
        self._commands = commands
        self._default_cmd_provider: CommandCompletionProvider = (
            default_command_provider if default_command_provider is not None else self._dummy_provider
        )
        self._general_provider: Optional[GeneralCompletionProvider] = general_provider


    # Optional general (non-command) provider.
    def set_general_provider(self, provider: Optional[GeneralCompletionProvider]) -> None:
        self._general_provider = provider

    def _dummy_provider(
        self, ui: UIState, document: Document, args: List[str], arg_prefix: str
    ) -> Iterable[Union[str, Completion]]:
        # Intentionally returns no suggestions for now.
        return []

    def _command_names(self) -> List[str]:
        # Use the names as registered (they include the leading '/' in this app).
        return [c.name for c in self._commands.list_commands()]

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor

        # General (non-command) completions when line doesn't start with '/'
        if not text.startswith("/"):
            if self._general_provider is None:
                return
            # Determine current word prefix (last whitespace-separated token)
            cursor_at_space = text.endswith(" ")
            tokens = text.split()
            current_word = "" if cursor_at_space else (tokens[-1] if tokens else "")
            for s in self._general_provider(document, current_word):
                if isinstance(s, Completion):
                    yield s
                else:
                    yield Completion(str(s), start_position=-len(current_word))
            return

        # Command-mode: nested completion
        cursor_at_space = text.endswith(" ")
        tokens = text.split()
        current_word = "" if cursor_at_space else (tokens[-1] if tokens else "")

        # If typing the command token (no space yet), suggest command names.
        if len(tokens) <= 1 and not cursor_at_space:
            prefix = current_word  # may be partial like '/he'
            for name in sorted(self._command_names()):
                if name.startswith(prefix):
                    yield Completion(name, start_position=-len(prefix))
            return

        # Otherwise, delegate to per-command provider (or dummy).
        cmd_name = tokens[0] if tokens else ""
        args = tokens[1:] if len(tokens) > 1 else []
        arg_prefix = "" if cursor_at_space else (current_word if len(tokens) >= 2 else "")

        cmd = self._commands.get(cmd_name)
        provider = (cmd.completer if cmd and cmd.completer else self._default_cmd_provider)
        for s in provider(self._ui, document, args, arg_prefix):
            if isinstance(s, Completion):
                yield s
            else:
                yield Completion(str(s), start_position=-len(arg_prefix))
