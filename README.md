# VectorOps Code

VectorOps Code is a desktop app for coding agents, much like Codex Desktop, but built around workflows you can write and customize in Python. Give an agent a task, review its proposed actions, and follow along as it reads files, runs commands, makes edits, and delegates work to helpers. Conversations, tool approvals, command output, and file changes are shown together.

Each conversation runs as a persistent workflow through [VectorOps Runner](https://github.com/vectorops/runner). Model calls, tool results, approvals, and child workflows are recorded, so interrupted work can resume using completed results rather than starting over. Long conversations can be summarized to fit the model’s context window while keeping the original execution history available for inspection.

The built-in coding agent is a starting point. You can write async Python workflows that combine agents, ordinary automation, and human decisions, then run them through the same interface. Model profiles, project skills, MCP tools, and Git worktrees let you adapt the setup to different tasks.

Code handles the interface; Runner handles execution and persistence; [Connect](https://github.com/vectorops/connect) handles model providers. A separately managed worker can keep working after you close the window. Saved execution history does not undo file changes or make every external action safe to repeat, so approvals and recovery still need care.

## Install from source

You need:

- Python 3.13 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
- Node.js 22 and npm to build the interface.
- Git to download the source.
- A native webview: WebKit on macOS, WebView2 on Windows, or a supported GTK/WebKit or Qt backend on Linux. See the [pywebview installation guide](https://pywebview.flowrl.com/guide/installation.html) for system packages.

Clone this repository. `uv` downloads the pinned Runner dependency and its dependencies, including Connect; separate checkouts are not needed.

```bash
git clone https://github.com/vectorops/code.git
cd code
uv sync --extra desktop
cd frontend
npm ci
npm run build
cd ..
```

Node.js is only needed to build the interface or work on it, not to run the built desktop application.

## Open a project

From the `code` directory, run:

```bash
uv run --extra desktop vectorops-code /path/to/your/project
```

Replace the path with the project you want to work on. Omit it to open the current directory. The desktop starts a local worker and connects automatically; no port or connection token setup is needed.

1. Open **Settings > LLM authentication** to add your provider credentials or sign in with a supported provider.
2. Open **Settings > LLM profiles** to choose a model.
3. Select **New conversation**, enter a task, and press **Enter**.
4. Review the responses and approve or deny tool requests as they appear.

Use **Advanced** to choose a different workflow and provide its inputs. See the [usage guide](docs/usage.md) for settings, shortcuts, and prompt history.

Closing the locally launched desktop stops its active workflows. To keep work running after closing the window, [run a separate worker](docs/connections.md).

## Your data and permissions

Runner manages workflow execution, provider credentials, and tool permissions. Review requested actions before approving them. Submitted prompt text is also saved locally for recall and search. See [data storage](docs/usage.md#data-storage) for locations and privacy details.

## Documentation

- [Documentation index](docs/index.md)
- [Using VectorOps Code](docs/usage.md)
- [Connecting to a separate worker](docs/connections.md)
- [Development and testing](docs/development/index.md)
- [Workflow example](examples/README.md)
- [VectorOps Runner](https://github.com/vectorops/runner) for workflow authoring and runner configuration
- [VectorOps Connect](https://github.com/vectorops/connect) for model-provider integration

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, checks, and pull request guidance.

## License

[MIT](LICENSE)
