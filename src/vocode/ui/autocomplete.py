from typing import Awaitable, Callable, Dict, List, Optional, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import UIState

AutoCompletionHandler = Callable[["UIState", Dict[str, Any]], Awaitable[List[str]]]


class AutoCompletionManager:
    def __init__(self, ui: "UIState") -> None:
        self._ui = ui
        self._registry: Dict[str, AutoCompletionHandler] = {}

    def register(self, name: str, handler: AutoCompletionHandler) -> None:
        self._registry[name] = handler

    def unregister(self, name: str) -> None:
        self._registry.pop(name, None)

    def get(self, name: str) -> Optional[AutoCompletionHandler]:
        return self._registry.get(name)
