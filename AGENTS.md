# Agent guide

This repository contains Runner Desktop, a Python desktop launcher and WebSocket adapter with a React interface. VectorOps Runner owns workflow execution, permissions, and persistence; VectorOps Connect provides model-provider integration.

## Layout and dependencies

- Python 3.13; dependencies and tool settings are in `pyproject.toml`.
- `src/runner_desktop/` contains the Python application; `tests/` contains its tests.
- `frontend/src/` contains React and TypeScript; `frontend/tests/` contains frontend tests.
- `examples/workflows/` contains examples, not application startup code.
- Development uses editable sibling checkouts at `../runner` and `../connect`.
- See `docs/development/index.md` for setup and build commands.

## Ownership and boundaries

- Keep each model, policy, and state transition with one authoritative owner.
- Call owners through explicit contracts. Do not inspect private structures or duplicate validation and decision rules.
- Adapters translate and delegate. Do not reimplement runner behavior in the desktop client.
- Pass narrow capabilities instead of whole runtime containers. Dependencies point from orchestration to owners, not the reverse.
- Keep registration in dedicated adapters loaded by explicit bootstrap. Reusable imports must not register unrelated capabilities.
- Preserve public contracts and deterministic replay. Do not rename or remove packet fields, discriminator values, or enum values without a migration.
- Keep new public fields optional and backward compatible.
- Never commit credentials, tokens, or local runtime data.
- Do not edit generated files under `src/runner_desktop/web/`; build them from frontend source.
- Preserve unrelated changes. Update tests only for intentional behavior changes.

## Python conventions

- Use precise types. Prefer `Optional[X]` to `X | None`, and avoid `Any` for structured data.
- Use Pydantic v2 for schemas, `default_factory` for mutable defaults, and discriminated unions for variant models.
- Normalize with a before model validator; check cross-field invariants with an after model validator.
- Use JSON-safe serialization, string enums, `Final` for constants, and `ClassVar` for registries.
- Do not use `getattr` or `hasattr` on strongly typed objects.
- Do not add comments unless requested; preserve existing comments. Use concise docstrings for complex functions.
- Group imports from the same module. Import no more than three symbols from a module; otherwise import the module and qualify names.
- Avoid blocking I/O in async handlers. Propagate timeouts and cancellation.
- Use `uv run` for all Python tools.

## Validation

- Format changed Python files with `uv run black PATH` and type-check with `uv run mypy PATH`.
- Run relevant Python tests with `uv run pytest`.
- From `frontend/`, run `npm test` and `npm run build` for frontend changes.
- Use the installed Prettier dependency to format changed frontend files only.
- Test schema validation, packet shapes, and service behavior at their owning boundaries.
- Report checks that failed or could not run. Do not fix unrelated failures.

## Documentation

Follow `CONTRIBUTING.md`. Keep the README focused on installation and first use. Write plain, task-oriented documentation without em dashes, promotional claims, plans, or implementation logs. Verify documented commands and settings against their source owners.
