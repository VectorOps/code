from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional, TYPE_CHECKING
from vocode.state import RunnerStatus

if TYPE_CHECKING:
    from vocode.ui.base import UIState


@dataclass
class CommandContext:
    ui: "UIState"
    out: Callable[[str], None]
    stop_toggle: Callable[[], Awaitable[None]]  # first call -> stop, second -> cancel
    request_exit: Callable[[], None]


@dataclass
class Command:
    name: str
    help: str
    usage: Optional[str]
    handler: Callable[[CommandContext, List[str]], Awaitable[None]]


_registry: Dict[str, Command] = {}


def register(name: str, help: str, usage: Optional[str] = None):
    def deco(func: Callable[[CommandContext, List[str]], Awaitable[None]]):
        _registry[name] = Command(name=name, help=help, usage=usage, handler=func)
        return func
    return deco


def list_commands() -> List[Command]:
    return sorted(_registry.values(), key=lambda c: c.name)


async def run(line: str, ctx: CommandContext) -> bool:
    parts = line.strip().split()
    if not parts:
        return True
    name = parts[0]
    cmd = _registry.get(name)
    if cmd is None:
        return False
    await cmd.handler(ctx, parts[1:])
    return True


# Built-in commands

@register("/help", "Show available commands")
async def _help(ctx: CommandContext, args: List[str]) -> None:
    ctx.out("Commands:")
    for c in list_commands():
        if c.usage:
            ctx.out(f"  {c.name} {c.usage} - {c.help}")
        else:
            ctx.out(f"  {c.name} - {c.help}")


@register("/workflows", "List available workflows")
async def _workflows(ctx: CommandContext, args: List[str]) -> None:
    names = ctx.ui.list_workflows()
    if not names:
        ctx.out("No workflows configured.")
        return
    for n in names:
        ctx.out(f"- {n}")


@register("/use", "Select and start a workflow", "<workflow>")
async def _use(ctx: CommandContext, args: List[str]) -> None:
    if not args:
        ctx.out("Usage: /use <workflow>")
        return
    name = args[0]
    try:
        await ctx.ui.use(name)
    except Exception as e:
        ctx.out(f"Failed to start workflow '{name}': {e}")


@register("/reset", "Reset current workflow from the beginning")
async def _reset(ctx: CommandContext, args: List[str]) -> None:
    try:
        await ctx.ui.reset()
    except Exception as e:
        ctx.out(f"Failed to reset: {e}")


@register("/stop", "Stop current run (Ctrl+C). Press twice to cancel.")
async def _stop(ctx: CommandContext, args: List[str]) -> None:
    await ctx.stop_toggle()


@register("/quit", "Exit the CLI")
async def _quit(ctx: CommandContext, args: List[str]) -> None:
    ctx.request_exit()


@register("/continue", "Continue execution if the runner is stopped")
async def _continue(ctx: CommandContext, args: List[str]) -> None:
    status = ctx.ui.status
    if status != RunnerStatus.stopped:
        if status == RunnerStatus.running:
            ctx.out("Run is already in progress.")
        elif status == RunnerStatus.waiting_input:
            ctx.out("Runner is waiting for input.")
        else:
            ctx.out(f"Cannot continue when status is '{status.value}'. Try /reset.")
        return
    try:
        await ctx.ui.restart()
    except Exception as e:
        ctx.out(f"Failed to continue: {e}")
