import asyncio
import argparse
import sys
from typing import Any, Dict, List, Optional
from enum import Enum, auto

from prompt_toolkit.application import Application
from prompt_toolkit.layout import Layout, HSplit
from prompt_toolkit.layout.containers import ConditionalContainer
from prompt_toolkit.filters import Condition
from prompt_toolkit.widgets import TextArea, Label
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from prompt_toolkit.layout.dimension import Dimension

import yaml

from vocode.graph.graph import Graph
from vocode.graph.models import Node, Edge
from vocode.runner.runner import Runner
from vocode.state import Message, Task, RunInput, RunnerStatus


class ChatUI:
    def __init__(self) -> None:
        self.history_lines: List[str] = [
            "Starting conversation. Type '/quit' to exit, '/stop' to gracefully stop, '/cancel' to cancel."
        ]
        self.streaming_text: Optional[str] = None
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self._input_enabled: bool = True
        self._input_visible: bool = True

        self.output = TextArea(
            text="\n".join(self.history_lines),
            read_only=True,
            focusable=False,
            focus_on_click=True,
            scrollbar=True,
            wrap_lines=True,
            height=Dimension(weight=1),
            style="class:output",
        )
        self.input = TextArea(
            prompt="Prompt: ",
            height=1,
            multiline=False,
            wrap_lines=False,
        )
        # Persist user input history to a file
        self._history = FileHistory(".chat_history")
        self.input.buffer.history = self._history

        self.input_container = ConditionalContainer(
            self.input,
            filter=Condition(lambda: self._input_visible),
        )

        def _accept(buf):
            text = buf.text
            if not self._input_enabled:
                # Ignore submissions while input is disabled
                return
            # Save to history (non-empty and not a slash-command)
            t = text.strip()
            if t and not t.startswith("/"):
                try:
                    self.input.buffer.history.append_string(text)
                except Exception:
                    pass
            # Clear input
            buf.document = Document("", cursor_position=0)
            # Hand over to event loop
            asyncio.create_task(self.queue.put(text))

        self.input.accept_handler = _accept

        kb = KeyBindings()

        @kb.add("c-c")
        def _(event):
            event.app.exit()

        root = HSplit([self.output, self.input_container])
        self.app = Application(
            layout=Layout(root),
            key_bindings=kb,
            style=Style.from_dict({"output": ""}),
            full_screen=False,
        )

    def _render(self) -> None:
        parts = list(self.history_lines)
        if self.streaming_text is not None:
            parts.append(self.streaming_text)
        text = "\n".join(parts)
        self.output.text = text
        # Scroll to bottom
        self.output.buffer.cursor_position = len(text)

    async def run(self) -> None:
        await self.app.run_async()

    async def exit(self) -> None:
        await self.app.exit_async()

    async def read_line(self) -> str:
        return await self.queue.get()

    def show_user(self, text: str) -> None:
        line = f"You: {text}"
        if self.history_lines and self.history_lines[-1] == line:
            # Avoid printing the same user line twice in a row
            return
        self.history_lines.append(line)
        self._render()

    def update_agent_stream(self, text: str) -> None:
        # Supports multiline partials naturally
        self.streaming_text = f"Agent: {text}"
        self._render()

    def finalize_agent(self, text: str) -> None:
        # Move streaming text into history (dedup last)
        self.streaming_text = None
        line = f"Agent: {text}"
        if self.history_lines and self.history_lines[-1] == line:
            # Avoid printing the same agent line twice in a row
            self._render()
            return
        self.history_lines.append(line)
        self._render()

    def info(self, text: str) -> None:
        self.history_lines.append(text)
        self._render()

    def set_prompt(self, text: str) -> None:
        self.input.prompt = text

    def enable_input(self, prompt: Optional[str] = None) -> None:
        self._input_enabled = True
        self.input.read_only = False
        self._input_visible = True
        if prompt is not None:
            self.set_prompt(prompt)

    def disable_input(self) -> None:
        self._input_enabled = False
        self.input.read_only = True
        self._input_visible = False
        self.set_prompt("")

    def set_status_prompt(self, status: str, node: Optional[str], suffix: str = "") -> None:
        node_part = f" {node}" if node else ""
        sep = " " if suffix else ""
        self.set_prompt(f"[{status}{node_part}]{sep}{suffix}")

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

async def handle_user_command(
    text: str,
    ui: ChatUI,
    runner: Optional[Runner] = None,
    *,
    allow_exit: bool = False,
    allow_new: bool = False,
) -> Optional[UserCmd]:
    """
    Parse and apply a user command. Returns the parsed command if recognized (even if acted upon),
    otherwise returns None. For QUIT when allow_exit=True, this will exit the UI and return UserCmd.QUIT.
    For STOP/CANCEL when a runner is provided, this will invoke the action and return the command.
    """
    cmd = parse_user_command(text)
    if cmd is UserCmd.NONE:
        return None

    if cmd is UserCmd.QUIT:
        if allow_exit:
            await ui.exit()
        else:
            if runner is not None:
                runner.stop()
        return cmd

    if cmd is UserCmd.STOP:
        if runner is not None:
            runner.stop()
        return cmd

    if cmd is UserCmd.CANCEL:
        if runner is not None:
            runner.cancel()
        return cmd

    if cmd is UserCmd.NEW and allow_new:
        return cmd

    if cmd is UserCmd.CONTINUE:
        return cmd

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


async def _run_conversation(graph: Graph, initial_text: Optional[str] = None) -> None:
    ui = ChatUI()

    async def conversation() -> None:
        nonlocal initial_text
        last_final_exec_id: Optional[str] = None
        while True:
            # 1) If there's no message yet, wait for user input
            if not initial_text:
                ui.enable_input()
                ui.set_status_prompt("await", None, "Enter prompt: ")
                user = (await ui.read_line()).strip()
                cmd = await handle_user_command(user, ui, runner=None, allow_exit=True)
                if cmd is UserCmd.QUIT:
                    return
                if not user or cmd is not None:
                    # ignore empty submissions and any command at this stage
                    continue
                initial_text = user
                ui.info("Input accepted. Starting execution...")
                ui.disable_input()

            # 2) Start a new runner flow with the provided message
            runner = Runner(graph, initial_messages=[Message(role="user", raw=initial_text)])
            task = Task()
            ui.show_user(initial_text)
            incoming: Optional[RunInput] = None

            driving = True
            current_node_name: Optional[str] = None
            while driving:
                agen = runner.run(task=task)
                try:
                    while True:
                        event = await (agen.asend(incoming) if incoming is not None else agen.__anext__())
                        incoming = RunInput()
                        current_node_name = event.node

                        # Explicit prompt when the runner requests more input for the node
                        if event.need_input:
                            while True:
                                ui.enable_input()
                                ui.set_status_prompt("await", event.node, "Enter message (or /stop | /cancel | /quit): ")
                                user = (await ui.read_line()).strip()
                                cmd = await handle_user_command(user, ui, runner)
                                if cmd in (UserCmd.QUIT, UserCmd.STOP, UserCmd.CANCEL):
                                    ui.disable_input()
                                    incoming = RunInput()  # let runner act on stop/cancel
                                    break
                                if not user:
                                    # Require a non-empty message when the node needs input
                                    continue
                                ui.show_user(user)
                                ui.disable_input()
                                incoming = RunInput(messages=[Message(role="user", raw=user)])
                                break
                            continue

                        if event.execution is not None:
                            msgs = event.execution.messages
                            last = msgs[-1] if msgs else None

                            if last and last.role == "agent":
                                text = last.raw

                                if not event.execution.is_complete:
                                    ui.disable_input()
                                    ui.update_agent_stream(text)
                                    continue
                                else:
                                    # Finalize the agent's message into history so multi-line responses are preserved
                                    ui.finalize_agent(text)

                                    # If executor finished without selecting an output, the runner will emit a need_input event next.
                                    # Don’t prompt here; just advance to that explicit need_input prompt.
                                    if event.execution.output_name is None:
                                        continue

                                    # Completed with multiple possible outputs: ask whether to proceed or loop explicitly.
                                    ui.info("Awaiting input: press Enter to proceed to the next node; type a message to loop this node with your input. Commands: /stop (pause), /cancel (cancel current), /quit (exit).")
                                    ui.enable_input()
                                    ui.set_status_prompt("await", event.node, "Action (Enter=next | message=loop | /stop | /cancel | /quit): ")
                                    user = (await ui.read_line()).strip()
                                    ui.disable_input()

                                    cmd = await handle_user_command(user, ui, runner)
                                    if cmd in (UserCmd.QUIT, UserCmd.STOP, UserCmd.CANCEL):
                                        continue
                                    if user:
                                        ui.show_user(user)
                                        incoming = RunInput(loop=True, messages=[Message(role="user", raw=user)])
                                    continue

                            continue

                        ui.disable_input()

                except StopAsyncIteration:
                    if runner.status in (RunnerStatus.stopped, RunnerStatus.canceled):
                        # Offer to continue the same execution
                        ui.info(f"Execution {runner.status}. Press Enter or type '/continue' to resume; type '/new' to start a new conversation; commands: /quit.")
                        ui.enable_input()
                        ui.set_status_prompt("resume", current_node_name, "Enter=/continue | /new | /quit: ")
                        raw = (await ui.read_line()).strip()
                        cmd = await handle_user_command(raw, ui, runner=None, allow_exit=True, allow_new=True)
                        if cmd is UserCmd.QUIT:
                            return
                        if cmd is UserCmd.NEW:
                            initial_text = None
                            driving = False
                            continue
                        # Treat Enter or '/continue' (or any other non-/new input) as resume
                        ui.disable_input()
                        incoming = None
                        continue
                    else:
                        ui.info(f"Conversation finished. Status: {runner.status}")
                        ui.enable_input()
                        ui.set_status_prompt("await", None, "Enter prompt: ")
                        # Start a new conversation next
                        initial_text = None
                        driving = False

    # Run UI and conversation concurrently
    conv_task = asyncio.create_task(conversation())
    try:
        await ui.run()
    finally:
        if not conv_task.done():
            conv_task.cancel()
            try:
                await conv_task
            except Exception:
                pass


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

