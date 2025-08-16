import asyncio
import argparse
import sys
from typing import Any, Dict, List, Optional
from enum import Enum, auto


import yaml

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

from vocode.graph.graph import Graph
from vocode.graph.models import Node, Edge
from vocode.runner.runner import Runner
from vocode.state import Message, Task, RunInput, RunnerStatus



class UserCmd(Enum):
    NONE = 0
    QUIT = auto()
    STOP = auto()
    CANCEL = auto()
    NEW = auto()
    CONTINUE = auto()

def parse_user_command(text: str) -> UserCmd:
    s = (text or "").strip().lower()
    if s in ("/quit", "/exit"):
        return UserCmd.QUIT
    if s == "/stop":
        return UserCmd.STOP
    if s == "/cancel":
        return UserCmd.CANCEL
    if s == "/new":
        return UserCmd.NEW
    if s == "/continue":
        return UserCmd.CONTINUE
    return UserCmd.NONE



def _pick_tool_data(data: Dict[str, Any], tool_name: Optional[str]) -> Optional[Dict[str, Any]]:
    # Support multiple YAML shapes:
    # 1) Flat: { nodes: [...], edges: [...] }
    # 2) Single tool: { tool: { name, nodes, edges } }
    # 3) Multiple tools: { tools: [ { name, nodes, edges }, ... ] }
    if "nodes" in data:
        return data
    if "tool" in data and isinstance(data["tool"], dict):
        return data["tool"]
    tools = data.get("tools")
    if isinstance(tools, list) and tools:
        if tool_name:
            for t in tools:
                if isinstance(t, dict) and t.get("name") == tool_name:
                    return t
        # default to first
        return tools[0]
    return None


def _build_graph_from_yaml(data: Dict[str, Any], tool_name: Optional[str]) -> Graph:
    tool_data = _pick_tool_data(data, tool_name)
    if tool_data is None:
        raise ValueError("YAML must contain either 'nodes'/'edges', 'tool', or 'tools' entries")
    nodes_raw = tool_data.get("nodes") or []
    edges_raw = tool_data.get("edges") or []
    if not isinstance(nodes_raw, list) or not isinstance(edges_raw, list):
        raise ValueError("'nodes' and 'edges' must be lists")
    nodes: List[Node] = [Node.from_obj(n) for n in nodes_raw]
    edges: List[Edge] = [Edge(**e) for e in edges_raw]
    return Graph.build(nodes=nodes, edges=edges)


_SESSION: Optional[PromptSession] = None

def _get_session() -> PromptSession:
    global _SESSION
    if _SESSION is None:
        try:
            hist = FileHistory(".simplecli_history")
        except Exception:
            hist = None  # fallback: no persistent history
        _SESSION = PromptSession(history=hist)
    return _SESSION

async def ainput(prompt: str = "") -> str:
    session = _get_session()
    try:
        return await session.prompt_async(prompt)
    except (EOFError, KeyboardInterrupt):
        # Treat Ctrl-D/Ctrl-C as a request to quit
        return "/quit"


async def _run_conversation(graph: Graph, initial_text: Optional[str] = None) -> None:
    def _print_user(text: str) -> None:
        print(f"You: {text}")

    def _print_agent(text: str) -> None:
        print(f"Agent: {text}")

    while True:
        # 1) Prompt for initial input if not provided
        if not initial_text:
            user = (await ainput("Enter prompt (or /quit): ")).strip()
            cmd = parse_user_command(user)
            if cmd is UserCmd.QUIT:
                return
            if not user or cmd is not UserCmd.NONE:
                # Empty or command other than /quit: reprompt
                continue
            initial_text = user

        # 2) Start a new runner with the provided message
        runner = Runner(graph, initial_messages=[Message(role="user", raw=initial_text)])
        task = Task()
        _print_user(initial_text)
        incoming: Optional[RunInput] = None
        driving = True

        while driving:
            agen = runner.run(task=task)
            try:
                while True:
                    event = await agen.asend(incoming)
                    incoming = RunInput()  # default: acknowledge and continue

                    # Need explicit user input for this node
                    if event.need_input:
                        while True:
                            user = (await ainput(f"[await {event.node}] Enter message (or /stop | /cancel | /quit): ")).strip()
                            cmd = parse_user_command(user)
                            if cmd is UserCmd.QUIT:
                                runner.stop()
                                break
                            if cmd is UserCmd.STOP:
                                runner.stop()
                                break
                            if cmd is UserCmd.CANCEL:
                                runner.cancel()
                                break
                            if not user:
                                continue
                            _print_user(user)
                            incoming = RunInput(messages=[Message(role="user", raw=user)])
                            break
                        continue

                    # Execution events: only print when is_complete
                    if event.execution is not None:
                        execn = event.execution
                        if not execn.is_complete:
                            continue

                        msgs = execn.messages
                        last = msgs[-1] if msgs else None
                        if last and last.role == "agent":
                            _print_agent(last.raw)

                        # Completed: ask user whether to proceed or loop this node
                        user = (await ainput("Action (Enter=next | message=loop | /stop | /cancel | /quit): ")).strip()
                        cmd = parse_user_command(user)
                        if cmd is UserCmd.QUIT:
                            runner.stop()
                            continue
                        if cmd is UserCmd.STOP:
                            runner.stop()
                            continue
                        if cmd is UserCmd.CANCEL:
                            runner.cancel()
                            continue

                        if user:
                            _print_user(user)
                            incoming = RunInput(loop=True, messages=[Message(role="user", raw=user)])

                        continue

            except StopAsyncIteration:
                if runner.status in (RunnerStatus.stopped, RunnerStatus.canceled):
                    # Offer to continue or start a new conversation
                    choice = (await ainput(f"Execution {runner.status}. Enter=/continue | /new | /quit: ")).strip()
                    cmd = parse_user_command(choice)
                    if cmd is UserCmd.QUIT:
                        return
                    if cmd is UserCmd.NEW:
                        initial_text = None
                        driving = False
                        continue
                    # Treat Enter or '/continue' as resume
                    incoming = None
                    continue
                else:
                    print(f"Conversation finished. Status: {runner.status}")
                    initial_text = None
                    driving = False


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Simple vocode CLI for running a YAML-defined conversation graph.")
    parser.add_argument("-f", "--file", required=True, help="Path to YAML file with nodes/edges (or tools).")
    parser.add_argument("-t", "--tool", default=None, help="Tool name to select when YAML provides multiple tools.")
    args = parser.parse_args(argv)

    with open(args.file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    graph = _build_graph_from_yaml(data, args.tool)

    asyncio.run(_run_conversation(graph))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

