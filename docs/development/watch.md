# Watch frontend and Python changes

Use the browser development command to run Vite and a restarting WebSocket worker together. This command requires a source checkout, not an installed wheel.

## Start

Run `uv sync` in the repository root and `npm ci` in `frontend/` first. No token configuration is needed for this local command.

```bash
uv run runner-desktop-dev --workspace /path/to/workspace
```

Open `http://127.0.0.1:5173`; the frontend connects automatically. The WebSocket endpoint defaults to `ws://127.0.0.1:8765`. Ctrl+C stops both development processes.

The launcher generates a fresh token per development session and passes it to its child processes. Vite delegates HTML rendering to a loopback-only, authenticated Python renderer. Jinja2 injects connection settings using `tojson` at request time, never into compiled JavaScript or files on disk. HTML is served with `Cache-Control: no-store`, with exact Host and Origin checks and cross-site requests rejected. The token stays unchanged across Python worker reloads. Remote and independently launched workers still require explicit credentials.

## Reload behavior

- Vite recompiles frontend modules and updates or reloads the browser when frontend files change. It does not write production assets; use `npm run build` for the packaged desktop frontend.
- Changes to Python files under this application's `src/`, sibling `runner/src/`, or sibling `connect/src/` restart the WebSocket worker after a short debounce. The browser reconnects automatically using the same token.
- Reload interrupts running workflows. On POSIX the worker receives SIGTERM and has 15 seconds to shut down; on Windows process termination is immediate. Persisted recovery follows the runner's recovery policy. Use a development workspace, not a production worker.
- If the worker fails to start because of a syntax error, its error appears in the terminal; fixing a watched Python file starts it again.
- Dependency changes still require `uv sync` and restarting this command. Vite transpilation does not type-check; `npm run build` runs TypeScript checking.

Use `--port` and `--frontend-port` to change ports. Add `--watch /another/python/source` to watch an additional existing directory; repeat it for multiple paths. This command serves the browser UI, not the native pywebview window.