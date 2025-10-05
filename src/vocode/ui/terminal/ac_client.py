from __future__ import annotations
import asyncio
from typing import Iterable, List, Union, Optional
from prompt_toolkit.document import Document
from prompt_toolkit.completion import Completion
from vocode.ui.rpc import RpcHelper
from vocode.ui.proto import UIPacketCompletionRequest, PACKET_COMPLETION_RESULT
from vocode.ui.terminal.completer import GeneralCompletionProvider
from vocode.ui.terminal.commands import CommandCompletionProvider

def make_canned_provider(rpc: RpcHelper, name: str, *, debounce: float = 0.3) -> CommandCompletionProvider:
    call_id = 0

    async def provider(_ui, document: Document, args: List[str], arg_prefix: str) -> Iterable[Union[str, Completion]]:
        nonlocal call_id
        call_id += 1
        my_id = call_id
        # Debounce: wait a bit; if a newer call arrived, cancel this one.
        await asyncio.sleep(debounce)
        if my_id != call_id:
            return []
        params = {
            "prefix": arg_prefix,
            "args": args,
            "text": document.text_before_cursor,
        }
        try:
            res = await rpc.call(
                UIPacketCompletionRequest(name=name, params=params),
                timeout=3.0,
            )
        except Exception:
            return []
        if not res or res.kind != PACKET_COMPLETION_RESULT or not res.ok:
            return []
        return res.suggestions

    return provider

def make_general_filelist_provider(
    rpc: RpcHelper, *, debounce: float = 0.3, limit: int = 20
) -> GeneralCompletionProvider:
    """
    General (non-command) completer backed by the server-side 'filelist' autocompleter.
    Debounced and awaited so first completion request yields suggestions immediately.
    """
    call_id: int = 0

    async def provider(document: Document, current_word: str):
        nonlocal call_id
        if not current_word:
            return []
        call_id += 1
        my_id = call_id
        # Debounce
        await asyncio.sleep(debounce)
        # If superseded, bail
        if my_id != call_id:
            return []
        params = {
            "prefix": current_word,
            "text": document.text_before_cursor,
            "limit": limit,
        }
        try:
            res = await rpc.call(
                UIPacketCompletionRequest(name="filelist", params=params),
                timeout=2.0,
            )
        except Exception:
            return []
        if not res or res.kind != PACKET_COMPLETION_RESULT or not getattr(res, "ok", False):
            return []
        return list(getattr(res, "suggestions", []))

    return provider
