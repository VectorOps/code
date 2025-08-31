from typing import Callable, Dict, List, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class Preprocessor:
    name: str
    description: str
    func: Callable[[str], str]


_registry: Dict[str, Preprocessor] = {}


def register_preprocessor(
    name: str,
    func: Callable[[str], str],
    description: str = "",
) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("Preprocessor name must be a non-empty string")
    _registry[name] = Preprocessor(name=name, description=description, func=func)


def get_preprocessor(name: str) -> Optional[Preprocessor]:
    return _registry.get(name)


def list_preprocessors() -> Dict[str, Preprocessor]:
    return dict(_registry)


def apply_preprocessors(names: List[str], text: str) -> str:
    result = text
    for n in names:
        pp = get_preprocessor(n)
        if pp is None:
            raise ValueError(f"Unknown preprocessor '{n}'")
        out = pp.func(result)
        if not isinstance(out, str):
            raise TypeError(f"Preprocessor '{n}' must return a string")
        result = out
    return result
