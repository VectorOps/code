# Workflow example

The [riddle workflow](workflows/riddles.py) asks a riddle and accepts guesses in a conversation. Incorrect guesses receive hints. A correct answer starts a new round with a fresh conversation history.

## Run it in a project

Configure a model and provider credentials using the [usage guide](../docs/usage.md#start-a-conversation).

With Runner's default workspace layout, project workflows live in `.voc/workflows/`. From this repository's root, copy the example into your test project:

```bash
mkdir -p /path/to/test-project/.voc/workflows
cp examples/workflows/riddles.py /path/to/test-project/.voc/workflows/riddles.py
uv run --extra desktop runner-desktop /path/to/test-project
```

If you have customized Runner's workspace or workflow directory, use that directory instead.

Choose **Advanced**, select **Riddle history test** (`riddles:workflow`), and start it without additional inputs. Reply to each riddle in the conversation. Use the stop control when you are finished; the example keeps starting new rounds until stopped.

This example makes model requests and uses your configured provider account. For writing your own workflows, see [VectorOps Runner](https://github.com/vectorops/runner).
