# Ensure built-in executors are registered on import
from .executors import llm  # noqa: F401
from .executors import input  # noqa: F401
