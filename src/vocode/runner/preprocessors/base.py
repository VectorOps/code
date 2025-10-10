from typing import Callable, Dict, List, Optional, Any, Sequence
from dataclasses import dataclass
from vocode.models import PreprocessorSpec

# Callback signature: accepts (project, spec, text)
PreprocessorFunc = Callable[[Any, PreprocessorSpec, str], str]


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
    preprocessors: Sequence[PreprocessorSpec], project: Any, text: str
) -> str:
    """
    Apply a sequence of preprocessors to the given text.
    - Accepts a sequence of PreprocessorSpec.
    - Passes the full PreprocessorSpec and project to the registered callback.
    """
    result = text
    for spec in preprocessors:
        pp = get_preprocessor(spec.name)
        if pp is None:
            raise ValueError(f"Unknown preprocessor '{spec.name}'")
        out = pp.func(project, spec, result)
        if not isinstance(out, str):
            raise TypeError(f"Preprocessor '{spec.name}' must return a string")
        result = out
    return result
