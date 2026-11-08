# Development

Follow the [source installation instructions](../../README.md#install-from-source). Only the `code` checkout is required. `uv sync` installs the development tools and downloads Runner and its dependencies from their pinned sources.

## Work on Runner or Connect locally

For cross-repository changes, optionally clone Runner and Connect beside `code`. Install the pinned dependencies first, then replace those two packages with editable checkouts:

```bash
uv sync
uv pip install --no-deps --editable ../runner --editable ../connect
uv run --no-sync vectorops-code-dev --workspace /path/to/test-project
```

Use `uv run --no-sync pytest` when testing local dependency changes. `--no-sync` keeps the editable installations instead of restoring the pinned packages. These overrides do not change the committed dependency pins. Run `uv sync` to return to the pinned dependencies. If local changes add dependencies, install those in the development environment too.

## Run while editing

From the repository root:

```bash
uv run vectorops-code-dev --workspace /path/to/test-project
```

Open `http://127.0.0.1:5173`. See [live reload](watch.md) for port options and restart behavior. Use a test project: Python reloads interrupt running workflows.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/vectorops_code/` | Desktop launcher, WebSocket server, and desktop-specific services |
| `frontend/src/` | React interface, connection handling, and display models |
| `tests/` | Python tests |
| `frontend/tests/` | Frontend tests |
| `examples/workflows/` | Workflows to copy into a test project |
| `.voc/` | Runner configuration for working on this repository |

Runner owns workflow execution, permissions, and persistence. This application provides the desktop interface and transport adapter. Keep runner behavior in the runner repository rather than implementing a second version here.

## Run checks

From the repository root:

```bash
uv run black --check src tests examples
uv run mypy src/vectorops_code examples
uv run pytest
```

From `frontend/`:

```bash
npm test
npm run build
```

The build command runs TypeScript checking before producing assets. The development server does not type-check.

Format changed Python files with `uv run black PATH`. For frontend changes, run `npx prettier --write PATH` from `frontend/` using the installed project dependency. Keep formatting changes limited to the files you touched.

## Build a Python package

Build the frontend first, then the Python distributions:

```bash
cd frontend
npm run build
cd ..
uv build
```

Vite writes assets to `src/vectorops_code/web/`. Those files are ignored by Git and included in the Python package. Rebuild them after frontend changes before launching the native desktop or packaging it.

## Add step or tool views

Step renderers are defined under `frontend/src/components/steps/`; tool formatters are under `frontend/src/components/tools/`. Use the existing registries rather than adding tool-specific conditions to the conversation screen.

Custom renderers receive a `debug` property through `StepProps` or `ToolProps`. Nested components can use `useDebug` or `DebugOnly` from `components/Diagnostics`. `DebugOnly` takes a render callback so hidden diagnostics are not evaluated. Keep useful results separate from diagnostic metadata.

`StepCard` treats its inspection action as diagnostic by default. Use `diagnostic={false}` when the action opens ordinary task details rather than internal records.

## Related

- [Contributing](../../CONTRIBUTING.md)
- [Desktop API reference](../reference/desktop-api.md)
- [Workflow example](../../examples/README.md)
