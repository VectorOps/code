from typing import Callable, Dict, List, Optional, Any, Sequence
from dataclasses import dataclass
from vocode.models import PreprocessorSpec

# Callback signature: accepts input text and optional options mapping
PreprocessorFunc = Callable[[str, Optional[Dict[str, Any]]], str]


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
    preprocessors: Sequence[PreprocessorSpec], text: str
) -> str:
    """
    Apply a sequence of preprocessors to the given text.
    - Accepts a sequence of PreprocessorSpec.
    - Passes spec.options to the registered callback.
    - If spec.prepend is true, the preprocessor output is prepended to the input
      rather than appended/transforming the whole text. To avoid double-appending in this mode,
      the preprocessor is invoked with an empty base string to produce just the injection text.
    """
    result = text
    for spec in preprocessors:
        pp = get_preprocessor(spec.name)
        if pp is None:
            raise ValueError(f"Unknown preprocessor '{spec.name}'")
        opts = dict(spec.options or {})
        if spec.prepend:
            # Produce injection only, then prepend to current result
            inj = pp.func("", opts)
            if not isinstance(inj, str):
                raise TypeError(f"Preprocessor '{spec.name}' must return a string")
            result = f"{inj}{result}"
        else:
            # Back-compat: let preprocessor transform/append to current result
            out = pp.func(result, opts)
            if not isinstance(out, str):
                raise TypeError(f"Preprocessor '{spec.name}' must return a string")
            result = out
    return result
