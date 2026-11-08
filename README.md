# Runner Desktop

Runner Desktop is a desktop interface for [VectorOps Runner](https://github.com/vectorops/runner). Start conversations and workflows in a project, respond to approval requests, and review messages, commands, and file changes in one place.

## Install from source

You need:

- Python 3.13 and [uv](https://docs.astral.sh/uv/getting-started/installation/).
- Node.js 22 and npm to build the interface.
- Git to download the source.
- A native webview: WebKit on macOS, WebView2 on Windows, or a supported GTK/WebKit or Qt backend on Linux. See the [pywebview installation guide](https://pywebview.flowrl.com/guide/installation.html) for system packages.

Clone all three repositories into the same parent directory. The dependency configuration expects these directory names:

```text
vectorops/
  code/
  connect/
  runner/
```

```bash
mkdir vectorops
cd vectorops
git clone https://github.com/vectorops/connect.git
git clone https://github.com/vectorops/runner.git
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
uv run --extra desktop runner-desktop /path/to/your/project
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
- [Using Runner Desktop](docs/usage.md)
- [Connecting to a separate worker](docs/connections.md)
- [Development and testing](docs/development/index.md)
- [Workflow example](examples/README.md)
- [VectorOps Runner](https://github.com/vectorops/runner) for workflow authoring and runner configuration
- [VectorOps Connect](https://github.com/vectorops/connect) for model-provider integration

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, checks, and pull request guidance.

## License

[MIT](LICENSE)
