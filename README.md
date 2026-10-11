# Runner desktop

A separate desktop client and WebSocket transport for VectorOps Runner. React renders one conversation tab per top-level workflow execution; pywebview owns only the native window. The Python transport layer has no GUI dependency and can also support a future TUI.

## Quick start

Requires Python 3.13, uv, and Node.js 22 for frontend development/builds. Keep this repository beside `runner/` and `connect/`; the development dependency configuration uses these sibling checkouts.

```bash
uv sync --extra desktop
cd frontend
npm ci
npm run build
cd ..
uv run --extra desktop runner-desktop desktop --workspace /path/to/workspace
```

Choose a workflow, inspect its input schema, and enter its JSON input to start a conversation. Select a tab to inspect persisted steps and answer pending text or JSON requests. Workflow definitions, permissions, validation, and persistence remain owned by the runner.

## Execution modes

| Mode | Command | Window close behavior |
| --- | --- | --- |
| In-process | `runner-desktop desktop --workspace ...` | Stops active workflows and joins the worker thread before exiting |
| Independent worker | `runner-desktop serve --workspace ... --origin ...` | Runs without a window |
| Attached desktop | `runner-desktop desktop --endpoint ws://127.0.0.1:8765` | Disconnects only; never shuts down the worker |

For an independent worker, generate a token with `uv run python -c "import secrets; print(secrets.token_urlsafe(32))"`. Set `RUNNER_DESKTOP_TOKEN` to that value in both terminals; do not put it in command-line arguments or commit it.

Start the worker in one terminal (or an OS service):

```bash
uv run runner-desktop serve --workspace /path/to/workspace --port 8765 --origin http://127.0.0.1:8766
```

Attach from another terminal:

```bash
uv run --extra desktop runner-desktop desktop --endpoint ws://127.0.0.1:8765
```

The desktop serves its assets at `http://127.0.0.1:8766` by default. Use `--asset-port` to change it and allow the matching origin on the worker. In-process mode generates a fresh token; attached mode requires `RUNNER_DESKTOP_TOKEN`. Tokens are handed to the frontend through a URL fragment, removed immediately from the address bar, and kept only in memory.

Independent mode attaches to a separately started worker; the desktop does not spawn or supervise that process. Use an OS service if the worker must survive terminal closure or user logout. Do not open two workers on the same workspace.

## Browser development

```bash
uv run runner-desktop serve --workspace /path/to/workspace --origin http://127.0.0.1:5173
```

In another terminal run `npm run dev` in `frontend/`, open `http://127.0.0.1:5173`, and enter the endpoint and token. Repeat `--origin` to allow multiple exact frontend origins. Connections without an allowed Origin are rejected, including non-browser clients unless they supply an allowed origin.

## Protocol and recovery

The WebSocket endpoint transports the runner's version-2 control packets unchanged as JSON text messages. Authentication is the first message, followed by requests, responses, errors, and subscription events. The adapter reuses the runner's authentication, limits, dispatch, and task ownership instead of maintaining a second implementation.

The frontend subscribes before refreshing persisted state, refreshes again when events arrive during a refresh, and polls every five seconds as a fallback. Reconnection reauthenticates and resubscribes. There is no durable event replay: sequence numbers are connection-local, and the history panel displays the latest 100 persisted steps. Start and input-response retries reuse their idempotency key when retried with unchanged content in the same mounted UI; commands are not automatically replayed after disconnect.

The current UI is a functional workflow inspector, not yet a rich chat renderer. It displays structured history and schema-driven JSON input rather than assuming all workflows accept a prompt. Specialized tool/diff views, paginated history navigation, OAuth interaction UI, durable replay, and native installers are not implemented.

## Platforms and security

The launcher uses cross-platform Python threads and loopback sockets; pywebview stays on the main GUI thread. macOS uses WebKit, Windows requires WebView2, and Linux requires a supported pywebview GTK/WebKit or Qt backend and its native packages. Install the appropriate backend prerequisites for your distribution. No Node.js runtime is needed after building the frontend. Native windows and installers still require validation on each OS.

The worker intentionally binds only `127.0.0.1`. For another machine, forward its port through SSH or put an authenticated WSS endpoint behind a correctly configured TLS reverse proxy; preserve the browser Origin and allow it explicitly. The client rejects unencrypted remote endpoints. A bearer token grants the runner's full control API, so share it only with trusted clients. There is no multi-user authorization layer.

## Development checks

```bash
uv run black src tests
uv run mypy src/runner_desktop
uv run pytest
cd frontend
npm run build
```

Build the frontend before building a Python wheel so the packaged `runner_desktop/web` assets are included. Source checkouts keep generated frontend assets out of version control. `AGENTS.md` is copied from the runner repository.