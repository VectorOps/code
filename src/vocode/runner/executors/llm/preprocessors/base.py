from typing import Callable, Dict, List, Optional, Any, Sequence
from dataclasses import dataclass
from vocode.models import PreprocessorSpec
from vocode.state import Message

# Callback signature: accepts (project, spec, text)
PreprocessorFunc = Callable[[Any, PreprocessorSpec, List[Message]], List[Message]]


@dataclass(frozen=True)
class Preprocessor:
    name: str
    description: str
    func: PreprocessorFunc


_registry: Dict[str, Preprocessor] = {}


def register_preprocessor(
    name: str,
    func: PreprocessorFunc,
    description: str = "",
) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("Preprocessor name must be a non-empty string")
    _registry[name] = Preprocessor(name=name, description=description, func=func)


def get_preprocessor(name: str) -> Optional[Preprocessor]:
    return _registry.get(name)


def list_preprocessors() -> Dict[str, Preprocessor]:
    return dict(_registry)


def apply_preprocessors(
    preprocessors: Sequence[PreprocessorSpec], project: Any, messages: List[Message]
) -> List[Message]:
    """
    Apply a sequence of preprocessors to a list of messages.
    """
    current_messages = list(messages)
    for spec in preprocessors:
        if preprocessor := get_preprocessor(spec.name):
            current_messages = preprocessor.func(project, spec, current_messages)
    return current_messages
