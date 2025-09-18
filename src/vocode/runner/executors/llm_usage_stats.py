from __future__ import annotations

from typing import AsyncIterator, Optional, Any

from vocode.runner.runner import Executor
from vocode.state import Message
from vocode.runner.models import (
    ReqPacket,
    ReqFinalMessage,
    ExecRunInput,
)


class LLMUsageStatsExecutor(Executor):
    # Node type this executor handles
    type = "llm_usage_stats"

    def __init__(self, config, project):
        super().__init__(config=config, project=project)

    async def run(self, inp: ExecRunInput) -> AsyncIterator[tuple[ReqPacket, Optional[Any]]]:
        # Read aggregates
        totals = self.project.llm_usage
        prompt = int(totals.prompt_tokens or 0)
        completion = int(totals.completion_tokens or 0)
        cost = float(totals.cost_dollars or 0.0)
        total_tokens = prompt + completion

        # Emit a final human-readable message
        text = (
            "LLM usage totals:\n"
            f"- Prompt tokens: {prompt:,}\n"
            f"- Completion tokens: {completion:,}\n"
            f"- Total tokens: {total_tokens:,}\n"
            f"- Cost: ${cost:,.4f}"
        )
        final = Message(role="agent", text=text)
        yield (ReqFinalMessage(message=final), inp.state)
        return
