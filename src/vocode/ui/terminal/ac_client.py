from __future__ import annotations
from typing import Iterable, List, Union
from prompt_toolkit.document import Document
from prompt_toolkit.completion import Completion
from vocode.ui.rpc import RpcHelper
from vocode.ui.proto import UIPacketCompletionRequest, PACKET_COMPLETION_RESULT
from vocode.ui.terminal.commands import CommandCompletionProvider

def make_canned_provider(rpc: RpcHelper, name: str) -> CommandCompletionProvider:
    async def provider(_ui, document: Document, args: List[str], arg_prefix: str) -> Iterable[Union[str, Completion]]:
        params = {"prefix": arg_prefix, "args": args, "text": document.text_before_cursor}
        try:
            res = await rpc.call(UIPacketCompletionRequest(name=name, params=params), timeout=3.0)
        except Exception:
            return []
        if not res or res.kind != PACKET_COMPLETION_RESULT or not res.ok:
            return []
        return res.suggestions
    return provider
