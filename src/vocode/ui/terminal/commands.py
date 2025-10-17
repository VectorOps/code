from dataclasses import dataclass
import shlex
from typing import (
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    TYPE_CHECKING,
    Iterable,
    Union,
)
from prompt_toolkit.document import Document
from prompt_toolkit.completion import Completion
from vocode.state import RunnerStatus
from vocode.ui.rpc import RpcHelper
from vocode.ui.proto import (
    UIPacketCompletionRequest,
    PACKET_COMPLETION_RESULT,
    UIPacketUIReload,
)

if TYPE_CHECKING:
    from vocode.ui.base import UIState

# Callback type used for per-command completions.
CommandCompletionProvider = Callable[
    ["UIState", Document, List[str], str],
    Union[
        Iterable[Union[str, Completion]], Awaitable[Iterable[Union[str, Completion]]]
    ],
]


@dataclass
class CommandContext:
    ui: "UIState"
    out: Callable[[str], None]
    stop_toggle: Callable[[], Awaitable[None]]  # first call -> stop, second -> cancel
    request_exit: Callable[[], None]
    rpc: RpcHelper


@dataclass
class Command:
    name: str
    help: str
    usage: Optional[str]
    handler: Callable[[CommandContext, List[str]], Awaitable[None]]
    completer: Optional[CommandCompletionProvider] = None


class Commands:
    def __init__(self) -> None:
        self._registry: Dict[str, Command] = {}

    def register(
        self,
        name: str,
        help: str,
        usage: Optional[str] = None,
        completer: Optional[CommandCompletionProvider] = None,
    ):
        def deco(func: Callable[[CommandContext, List[str]], Awaitable[None]]):
            self._registry[name] = Command(
                name=name, help=help, usage=usage, handler=func, completer=completer
            )
            return func

        return deco

    def list_commands(self) -> List[Command]:
        return sorted(self._registry.values(), key=lambda c: c.name)

    def unregister(self, name: str) -> None:
        if name in self._registry:
            del self._registry[name]

    def get(self, name: str) -> Optional[Command]:
        return self._registry.get(name)

    async def run(self, line: str, ctx: CommandContext) -> bool:
        s = line.strip()
        if not s:
            return True
        try:
            # Shell-like parsing: supports quotes and backslash escaping.
            # Disable comments to avoid treating '#' as a comment introducer.
            lexer = shlex.shlex(s, posix=True)
            lexer.whitespace_split = True
            lexer.commenters = ""
            parts = list(lexer)
        except ValueError as e:
            # Unbalanced quotes or similar parsing error.
            await ctx.out(f"Parse error: {e}")
            return True

        if not parts:
            return True
        name = parts[0]
        cmd = self._registry.get(name)
        if cmd is None:
            return False
        args = parts[1:]
        await cmd.handler(ctx, args)
        return True


# Built-in command handlers (registration happens via register_default_commands)


async def _workflows(ctx: CommandContext, args: List[str]) -> None:
    names: List[str] = []
    try:
        res = await ctx.rpc.call(
            UIPacketCompletionRequest(name="workflow_list", params={}), timeout=3.0
        )
        if res and res.kind == PACKET_COMPLETION_RESULT and res.ok:
            names = list(res.suggestions)
        else:
            # Fallback to direct API if RPC not available or failed
            names = ctx.ui.list_workflows()
    except Exception:
        # Fallback to direct API on any RPC error
        names = ctx.ui.list_workflows()

    if not names:
        await ctx.out("No workflows configured.")
        return
    for n in names:
        await ctx.out(f"- {n}")


async def _use(ctx: CommandContext, args: List[str]) -> None:
    if not args:
        await ctx.out("Usage: /use <workflow>")
        return
    name = args[0]
    try:
        await ctx.ui.use(name)
    except Exception as e:
        await ctx.out(f"Failed to start workflow '{name}': {e}")


async def _run(ctx: CommandContext, args: List[str]) -> None:
    if not args:
        await ctx.out("Usage: /run <workflow>")
        return
    name = args[0]
    try:
        await ctx.ui.use(name)
    except Exception as e:
        await ctx.out(f"Failed to run workflow '{name}': {e}")


async def _reset(ctx: CommandContext, args: List[str]) -> None:
    try:
        await ctx.ui.reset()
    except Exception as e:
        await ctx.out(f"Failed to reset: {e}")


async def _stop(ctx: CommandContext, args: List[str]) -> None:
    await ctx.stop_toggle()


async def _quit(ctx: CommandContext, args: List[str]) -> None:
    ctx.request_exit()


async def _continue(ctx: CommandContext, args: List[str]) -> None:
    status = ctx.ui.status
    if status != RunnerStatus.stopped:
        if status == RunnerStatus.running:
            await ctx.out("Run is already in progress.")
        elif status == RunnerStatus.waiting_input:
            await ctx.out("Runner is waiting for input.")
        else:
            await ctx.out(
                f"Cannot continue when status is '{status.value}'. Try /reset."
            )
        return
    try:
        await ctx.ui.restart()
    except Exception as e:
        await ctx.out(f"Failed to continue: {e}")


async def _reload(ctx: CommandContext, args: List[str]) -> None:
    """
    Reload project configuration and restart UI/project state.
    To avoid accidental reloads, require an explicit confirmation argument.
    """
    confirm = args[0].strip().lower() if args else ""
    if confirm not in ("confirm", "yes", "y"):
        await ctx.out("This will reload the project. Run '/reload confirm' to proceed.")
        return
    try:
        await ctx.rpc.call(UIPacketUIReload(), timeout=None)
        await ctx.out("Reloaded project.")
    except Exception as e:
        await ctx.out(f"Reload failed: {e}")


def register_default_commands(
    commands: Commands,
    ui: "UIState",
    ac_factory: Optional[Callable[[str], CommandCompletionProvider]] = None,
) -> Commands:
    # Help must access the instance to list commands; define it in this scope.
    @commands.register("/help", "Show available commands")
    async def _help(ctx: CommandContext, args: List[str]) -> None:
        await ctx.out("Commands:")
        for c in commands.list_commands():
            if c.usage:
                await ctx.out(f"  {c.name} {c.usage} - {c.help}")
            else:
                await ctx.out(f"  {c.name} - {c.help}")

    # Register the rest of the built-ins against this instance.
    commands.register("/workflows", "List available workflows")(_workflows)

    commands.register("/reset", "Reset current workflow from the beginning")(_reset)
    commands.register("/stop", "Stop current run (Ctrl+C). Press twice to cancel.")(
        _stop
    )
    commands.register("/quit", "Exit the CLI")(_quit)
    commands.register("/continue", "Continue execution if the runner is stopped")(
        _continue
    )
    commands.register(
        "/reload",
        "Reload config and restart project state",
        "[confirm]",
    )(_reload)

    commands.register(
        "/run",
        "Run a workflow",
        "<workflow>",
        completer=(ac_factory("workflow_list") if ac_factory else None),
    )(_run)

    return commands
