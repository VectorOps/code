# Contributing

Use issues to report bugs or discuss changes, and pull requests to propose fixes. Include the steps needed to reproduce a problem and the relevant error output. Remove credentials, connection tokens, and private project content before sharing logs or screenshots.

## Set up

Follow [installation from source](README.md#install-from-source). The [development guide](docs/development/index.md) covers live reload, tests, formatting, and packaging.

## Prepare a change

- Keep each pull request focused on one change.
- Add or update tests for changed behavior.
- Run the relevant [development checks](docs/development/index.md#run-checks) and report any checks you could not run.
- Update documentation when commands, settings, or user-visible behavior change.
- Do not commit generated frontend assets, local data, or credentials.

Runner owns workflow execution and provider configuration. Changes to those behaviors belong in [Runner](https://github.com/vectorops/runner) or [Connect](https://github.com/vectorops/connect), not a duplicate implementation here.

## Write documentation

Write for the person trying to complete a task. Start with what to do, provide a small working example, and explain the result. Use plain language and the labels shown in the application.

Keep installation and first use in the README. Put detailed usage, connection instructions, development guidance, and API reference under `docs/`. Link to the page that owns a topic instead of copying it into several places.

Document current behavior, not delivery plans, implementation logs, or speculative features. Avoid promotional claims, jargon, and em dashes. Check relative links and commands before submitting.
