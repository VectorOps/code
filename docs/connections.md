# Connecting to a separate worker

A worker runs workflows for one project. Start it separately when you want work to continue after closing the desktop window. For ordinary local use, use the [single-command launch](../README.md#open-a-project) instead.

## Start the worker

Run from the `code` checkout on the machine containing your project:

```bash
uv run vectorops-code serve /path/to/your/project
```

The worker allocates an available loopback WebSocket port and prints its endpoint. It also generates a persistent connection token and prints the token file's path, not the secret. Use `--port PORT` if a tunnel or reverse proxy requires a stable worker port. `--workspace PATH` remains supported instead of the positional project path.

## Connect the desktop

On the desktop machine, run:

```bash
uv run --extra desktop vectorops-code connect
```

Enter the remote address and WebSocket port when prompted. A non-loopback hostname uses WSS, so it requires a TLS reverse proxy. For SSH forwarding, enter the local tunnel address and port instead. You can also supply an address directly:

```bash
uv run --extra desktop vectorops-code connect wss://worker.example.com/runner
uv run --extra desktop vectorops-code connect 127.0.0.1 --port 18765
```

The first connection prompts for the worker's token without echoing it and saves it locally for that endpoint. Retrieve the token from the worker's printed token file using a trusted channel, such as an SSH session. Do not put the token in command arguments, source control, or bug reports. No local project is needed or created.

The desktop allocates its frontend HTTP port automatically. Multiple windows can attach without competing for a fixed frontend port. Different local projects can also run concurrently with `vectorops-code /path/to/project`; each gets its own automatically allocated worker and frontend ports. Ports are selected by the operating system, not by probing a port and releasing it before startup.

Closing this desktop only disconnects it. Stop the worker in its terminal with Ctrl+C. Use an operating-system service to keep it running independently of a terminal session. Do not start two workers for the same project.

## Token storage and overrides

Authentication uses one shared bearer token. An automatically generated worker token and a user-provided token use the same protocol; the client must have the same value as the worker.

| Launch mode | Token storage |
| --- | --- |
| `vectorops-code PROJECT` | New random token in memory for that local launch; not saved |
| `vectorops-code serve PROJECT` | Worker user's `~/.config/vectorops-code/workers/<project-hash>.token`, reused on restart |
| `vectorops-code connect ADDRESS` | Desktop user's `~/.config/vectorops-code/connections/<endpoint-hash>.token`, saved after the first prompt |

`VECTOROPS_CODE_CONFIG_DIR` overrides the Code configuration directory on each machine. Worker file names hash the resolved project path; connection file names hash the endpoint URL. These files are separate from Runner configuration and provider credentials. They contain plaintext secrets, protected with owner-only file permissions on POSIX systems, not an OS keychain. Do not share or commit this directory.

To supply credentials without a prompt, use `--token-file PATH` or set `VECTOROPS_CODE_TOKEN` in the process environment. Explicit token files take precedence over environment variables; environment variables take precedence over default saved files. `RUNNER_DESKTOP_TOKEN` is accepted when `VECTOROPS_CODE_TOKEN` is unset. Environment-provided tokens are not saved. `serve --token-file PATH` creates a token if the file does not exist; `connect --token-file PATH` requires an existing file. On POSIX systems, supplied token files must be owned by the current user with no group or other permissions (`chmod 600 PATH`).

For a user-provided token, generate at least 32 random bytes, for example with Python's `secrets.token_urlsafe(32)`. The Runner validates token syntax and performs authentication. A syntactically valid token entered at the prompt is saved before the connection is authenticated. If it is incorrect or rotated, remove the endpoint's saved file to prompt again, or supply an updated `--token-file`. Remove the worker's generated token file while it is stopped to generate a replacement on restart, then update clients. Changing a tunnel endpoint creates a different client credential entry.

## SSH forwarding

The CLI does not install remote workers or manage SSH tunnels. Start a worker on the remote machine, then forward its printed port through SSH:

```bash
ssh -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:18765:127.0.0.1:WORKER_PORT user@host
```

Replace `WORKER_PORT` with the worker's printed port. The example uses `18765` for the local tunnel; choose another available port for another tunnel. Keep SSH host-key verification enabled. Connect with `vectorops-code connect 127.0.0.1:18765`. Only the SSH tunnel needs a manually chosen local port; no frontend port or origin configuration is required.

## Connection security

The worker binds to `127.0.0.1`. A connection token grants access to the runner's full control API, including workflow actions. Share it only with trusted clients. There is no separate authorization level for each user.

For access from another machine, use SSH port forwarding or a TLS reverse proxy with a `wss://` endpoint. A reverse proxy must preserve the browser Origin header. The desktop rejects unencrypted remote `ws://` endpoints; an SSH tunnel can expose the forwarded connection on local loopback. SSH encrypts the network traffic but does not replace token authentication.

By default, a separate worker accepts frontend origins of the form `http://127.0.0.1:PORT`, allowing the native desktop's automatically allocated frontend port. This trusts loopback frontend origins, not arbitrary websites. Missing origins, `null`, other hosts, and nonmatching origins are rejected. Token authentication is still mandatory. Local managed workers allow only their own desktop's exact origin.

For a stricter allowlist or browser development, supply `serve --origin http://127.0.0.1:8766` and attach with `desktop --endpoint URL --asset-port 8766`. Repeat `--origin` for multiple exact origins. Explicit origins replace the default loopback-origin policy; older workers using exact origins still require matching `--asset-port` values.

The attached desktop does not start, discover, or reconfigure the worker. Connection tokens passed to the interface through a URL fragment are removed from the address bar and kept in memory.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| Address already in use | Omit explicit worker/frontend ports to allocate available ones. For SSH, choose a free local forwarding port. Explicit ports do not fall back to another port. |
| Authentication fails | Supply the same token as the worker. Replace a stale saved client token or use `--token-file`. Restart the worker after changing its token. |
| Origin rejected | With explicit origins or an older worker, match the complete origin, including scheme, hostname, and port. `localhost` and `127.0.0.1` are different origins. |
| Desktop assets are missing | Run `npm ci` and `npm run build` in `frontend/`. |
| Native window cannot open | Check the [pywebview system requirements](https://pywebview.flowrl.com/guide/installation.html) for your platform. |
| Worker exits unexpectedly | Read the error in the terminal before starting another worker for the project. |

For a locally launched desktop, worker startup waits up to 30 seconds. Shutdown allows 15 seconds for cleanup, then requests cancellation and waits up to another 15 seconds. If shutdown times out, exit the application before reopening the project; the launcher does not forcibly kill a stuck worker thread.

## Related

- [Usage and stored data](usage.md)
- [Browser development](development/watch.md)
- [Transport reference](reference/desktop-api.md#transport)
