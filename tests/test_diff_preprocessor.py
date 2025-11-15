import pytest
from typing import List
from unittest.mock import patch

from vocode.runner.executors.llm.preprocessors.base import get_preprocessor
from vocode.runner.executors.llm.preprocessors import diff as diff_mod
from vocode.models import PreprocessorSpec, Mode
from vocode.state import Message

class DummyProject:
    def __init__(self):
        pass

@pytest.fixture
def base_messages() -> List[Message]:
    return [
        Message(role="system", text="system prompt"),
        Message(role="user", text="hello"),
    ]

def test_diff_preprocessor_does_not_reinject(base_messages):
    """Test diff preprocessor does not reinject content."""
    pp = get_preprocessor("diff")
    assert pp is not None

    spec = PreprocessorSpec(name="diff", options={"format": "test_format"})
    
    with patch(
        "vocode.runner.executors.llm.preprocessors.diff.get_supported_formats",
        return_value=("test_format",),
    ), patch(
        "vocode.runner.executors.llm.preprocessors.diff.get_system_instruction",
        return_value=" DIFF",
    ):
        # First call injects the content
        out_messages = pp.func(DummyProject(), spec, base_messages)
        assert out_messages[0].text == "system prompt DIFF"

        # Second call should not change the messages
        final_messages = pp.func(DummyProject(), spec, out_messages)
        assert final_messages[0].text == "system prompt DIFF"