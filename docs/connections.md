# Connecting to a separate worker

A worker runs workflows for one project. Start it separately when you want work to continue after closing the desktop window. For ordinary local use, use the [single-command launch](../README.md#open-a-project) instead.

## Start the worker

Run these commands from the `code` checkout. Generate a token:

```bash
uv run python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Set `VECTOROPS_CODE_TOKEN` to that value in both the worker's environment and the desktop's environment. Use your shell's environment-variable syntax. Do not include the token in application command-line arguments, source files, or bug reports. Existing setups using `RUNNER_DESKTOP_TOKEN` are still accepted when `VECTOROPS_CODE_TOKEN` is unset.

Start the worker in one terminal:

```bash
uv run vectorops-code serve --workspace /path/to/your/project --port 8765 --origin http://127.0.0.1:8766
```

Open the desktop from another terminal:

```bash
uv run --extra desktop vectorops-code desktop --endpoint ws://127.0.0.1:8765 --asset-port 8766
```

The desktop's asset port must match the origin allowed by the worker. Repeat `--origin` to allow more than one exact frontend origin. Connections without an allowed Origin header are rejected, including non-browser clients.

Closing this desktop only disconnects it. Stop the worker in its terminal with Ctrl+C. Use an operating-system service to keep it running independently of a terminal session. Do not start two workers for the same project.

## Connection security

The worker binds to `127.0.0.1`. A connection token grants access to the runner's full control API, including workflow actions. Share it only with trusted clients. There is no separate authorization level for each user.

For access from another machine, use SSH port forwarding or a TLS reverse proxy with a `wss://` endpoint. A reverse proxy must preserve the browser Origin header, which must be explicitly allowed by the worker. The desktop rejects unencrypted remote `ws://` endpoints; an SSH tunnel can expose the forwarded connection on local loopback.

The attached desktop does not start, discover, or reconfigure the worker. Connection tokens passed to the interface through a URL fragment are removed from the address bar and kept in memory.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| Address already in use | Choose unused ports and update the endpoint, asset port, and allowed origin together. Explicit ports do not fall back to another port. |
| Authentication fails | Both processes must receive the same `VECTOROPS_CODE_TOKEN`. Restart them after changing it. |
| Origin rejected | Match the complete origin, including scheme, hostname, and port. `localhost` and `127.0.0.1` are different origins. |
| Desktop assets are missing | Run `npm ci` and `npm run build` in `frontend/`. |
| Native window cannot open | Check the [pywebview system requirements](https://pywebview.flowrl.com/guide/installation.html) for your platform. |
| Worker exits unexpectedly | Read the error in the terminal before starting another worker for the project. |

For a locally launched desktop, worker startup waits up to 30 seconds. Shutdown allows 15 seconds for cleanup, then requests cancellation and waits up to another 15 seconds. If shutdown times out, exit the application before reopening the project; the launcher does not forcibly kill a stuck worker thread.

## Related

- [Usage and stored data](usage.md)
- [Browser development](development/watch.md)
- [Transport reference](reference/desktop-api.md#transport)
