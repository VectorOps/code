# Live reload

Run the interface in a browser while editing React and Python source. This command requires a source checkout with the [development dependencies](index.md) installed.

## Start

Run `uv sync` in the repository root and `npm ci` in `frontend/`, then start from the repository root:

```bash
uv run vectorops-code-dev --workspace /path/to/test-project
```

Open `http://127.0.0.1:5173`. The interface connects automatically to `ws://127.0.0.1:8765`. The launcher creates a connection token; no manual token or origin configuration is needed. Ctrl+C stops both processes.

## Reload behavior

- Frontend edits update or reload the browser through Vite. They do not rebuild the native desktop's assets.
- Python edits under this application's `src/` restart the worker. Sibling `../runner/src/` and `../connect/src/` directories are also watched when present; use [editable overrides](index.md#work-on-runner-or-connect-locally) to run their code. The browser reconnects with the same token.
- Reload interrupts running workflows. On POSIX the worker receives SIGTERM and has 15 seconds to stop; on Windows it is terminated immediately. Use a test project, not a worker handling important tasks.
- A Python syntax error appears in the terminal. Fixing a watched file starts the worker again.
- Dependency changes require `uv sync` and a restart of the development command.

## Options

| Option | Purpose |
| --- | --- |
| `--port` | Change the worker port from `8765` |
| `--frontend-port` | Change the Vite port from `5173` |
| `--watch PATH` | Watch another existing Python source directory; repeat for multiple paths |

Ports must be available; the launcher does not silently select replacements. This command runs the browser interface, not the native window.

## Run the processes separately

Use the [token setup](../connections.md#start-the-worker), then start the worker with the Vite origin:

```bash
uv run vectorops-code serve --workspace /path/to/test-project --port 8765 --origin http://127.0.0.1:5173 --development
```

In another terminal, run `npm run dev` from `frontend/`. Open `http://127.0.0.1:5173` and enter the worker endpoint and token. This setup does not restart Python automatically.

Run the [development checks](index.md#run-checks) before submitting changes.
