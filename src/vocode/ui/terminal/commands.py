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


class Commands:
    def __init__(self) -> None:
        self._registry: Dict[str, Command] = {}

    def register(self, name: str, help: str, usage: Optional[str] = None):
        def deco(func: Callable[[CommandContext, List[str]], Awaitable[None]]):
            self._registry[name] = Command(
                name=name, help=help, usage=usage, handler=func
            )
            return func
        return deco

    def list_commands(self) -> List[Command]:
        return sorted(self._registry.values(), key=lambda c: c.name)

    async def run(self, line: str, ctx: CommandContext) -> bool:
        parts = line.strip().split()
        if not parts:
            return True
        name = parts[0]
        cmd = self._registry.get(name)
        if cmd is None:
            return False
        await cmd.handler(ctx, parts[1:])
        return True


# Built-in command handlers (registration happens via register_default_commands)

async def _workflows(ctx: CommandContext, args: List[str]) -> None:
    names = ctx.ui.list_workflows()
    if not names:
        ctx.out("No workflows configured.")
        return
    for n in names:
        ctx.out(f"- {n}")


async def _use(ctx: CommandContext, args: List[str]) -> None:
    if not args:
        ctx.out("Usage: /use <workflow>")
        return
    name = args[0]
    try:
        await ctx.ui.use(name)
    except Exception as e:
        ctx.out(f"Failed to start workflow '{name}': {e}")


async def _reset(ctx: CommandContext, args: List[str]) -> None:
    try:
        await ctx.ui.reset()
    except Exception as e:
        ctx.out(f"Failed to reset: {e}")


async def _stop(ctx: CommandContext, args: List[str]) -> None:
    await ctx.stop_toggle()


async def _quit(ctx: CommandContext, args: List[str]) -> None:
    ctx.request_exit()


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


def register_default_commands(commands: Commands) -> Commands:
    # Help must access the instance to list commands; define it in this scope.
    @commands.register("/help", "Show available commands")
    async def _help(ctx: CommandContext, args: List[str]) -> None:
        ctx.out("Commands:")
        for c in commands.list_commands():
            if c.usage:
                ctx.out(f"  {c.name} {c.usage} - {c.help}")
            else:
                ctx.out(f"  {c.name} - {c.help}")

    # Register the rest of the built-ins against this instance.
    commands.register("/workflows", "List available workflows")(_workflows)
    commands.register("/use", "Select and start a workflow", "<workflow>")(_use)
    commands.register("/reset", "Reset current workflow from the beginning")(_reset)
    commands.register("/stop", "Stop current run (Ctrl+C). Press twice to cancel.")(_stop)
    commands.register("/quit", "Exit the CLI")(_quit)
    commands.register("/continue", "Continue execution if the runner is stopped")(_continue)
    return commands
