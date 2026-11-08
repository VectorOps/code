# Development

Follow the [source installation instructions](../../README.md#install-from-source) to place `code`, `runner`, and `connect` beside one another and install dependencies. Runner is installed as an editable dependency from `../runner`. Keep both sibling source trees available for type checking and live reload.

Runner selects the Connect dependency used at runtime. Cloning Connect alone does not override that selection; check the resolved dependency before testing local Connect changes.

## Run while editing

From the repository root:

```bash
uv run runner-desktop-dev --workspace /path/to/test-project
```

Open `http://127.0.0.1:5173`. See [live reload](watch.md) for port options and restart behavior. Use a test project: Python reloads interrupt running workflows.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/runner_desktop/` | Desktop launcher, WebSocket server, and desktop-specific services |
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
uv run mypy src/runner_desktop examples
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

Vite writes assets to `src/runner_desktop/web/`. Those files are ignored by Git and included in the Python package. Rebuild them after frontend changes before launching the native desktop or packaging it.

## Add step or tool views

Step renderers are defined under `frontend/src/components/steps/`; tool formatters are under `frontend/src/components/tools/`. Use the existing registries rather than adding tool-specific conditions to the conversation screen.

Custom renderers receive a `debug` property through `StepProps` or `ToolProps`. Nested components can use `useDebug` or `DebugOnly` from `components/Diagnostics`. `DebugOnly` takes a render callback so hidden diagnostics are not evaluated. Keep useful results separate from diagnostic metadata.

`StepCard` treats its inspection action as diagnostic by default. Use `diagnostic={false}` when the action opens ordinary task details rather than internal records.

## Related

- [Contributing](../../CONTRIBUTING.md)
- [Desktop API reference](../reference/desktop-api.md)
- [Workflow example](../../examples/README.md)
