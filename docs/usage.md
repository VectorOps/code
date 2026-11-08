# Using Runner Desktop

Open a project with the command in the [README](../README.md#open-a-project). Each conversation represents a workflow running in that project.

## Start a conversation

Configure provider access in **Settings > LLM authentication** and choose a model in **Settings > LLM profiles**. Provider access uses API keys or browser sign-in, depending on the provider.

Select **New conversation** to start the default workflow. Enter a task, then press **Enter** to send it. Use **Advanced** to select another workflow and fill in its input fields. The advanced form also offers **Edit as JSON** for structured inputs.

Tool requests that need permission appear in the conversation. Review the action and its scope before approving it. Manage permission rules in **Settings > Tool permissions**.

Stopping a workflow does not undo actions it has already completed. Closing a locally launched desktop stops its workflows; closing a desktop attached to a [separate worker](connections.md) only disconnects the window.

## Write and recall messages

| Shortcut | Action |
| --- | --- |
| Enter | Send when sending is available |
| Shift, Alt, Ctrl, or Cmd + Enter | Insert a newline |
| Ctrl+J | Insert a newline |
| Up on the first visual line | Recall an older submitted prompt |
| Down on the last visual line | Recall a newer prompt, then restore the draft |
| Ctrl+R or Cmd+R | Search prompt history from the message box or conversation |
| Ctrl+N or Cmd+N | Start the default workflow |
| Ctrl+Shift+N or Cmd+Shift+N | Open advanced workflow selection |

Enter does not stop or resume a workflow. Multiline paste stays multiline, and confirming text with an input method does not submit it.

Prompt recall uses submitted text across projects and conversations on the same host account. The search dialog has **Global**, **This project**, and **This chat** filters. Search ignores letter case and treats the query as literal text. Repeated text appears once at its latest position within the selected scope.

Choose a result to restore its text without sending it or importing attachments. Press **Escape** to keep the current draft. Ctrl/Cmd+R opens search rather than refreshing the page; in browser development, Ctrl/Cmd+Shift+R remains available for a hard refresh.

## Settings

| Setting | Where to find it |
| --- | --- |
| Light, dark, or system appearance | Settings > Visual layout |
| Default workflow and displayed history limits | Settings > Conversations |
| Model selection | Settings > LLM profiles |
| Provider credentials | Settings > LLM authentication |
| Workflow configuration | Settings > Workflows |
| Tool permission rules | Settings > Tool permissions |
| Isolated working copies | Settings > Worktrees |
| Diagnostic details | Settings > Developer tools |

Conversation history settings control how many execution steps the interface loads and keeps in memory, not how much history Runner stores on disk. The defaults load 1,000 steps initially and 200 per request for earlier steps, with a retention ceiling of 2,000 and a trim target of 1,000. Some execution steps are hidden from the conversation view, so these are not message counts.

### Diagnostic details

Under **Settings > Developer tools > Debug interface**, choose:

- **Auto**: show diagnostics for a development worker and hide them for a normal worker.
- **On**: always show diagnostics.
- **Off**: hide diagnostics.

Diagnostics include raw records, configuration JSON, and internal identifiers. Commands, results, diffs, errors, and approval controls remain available when diagnostics are hidden. This setting changes the display, not access permissions.

## Data storage

Desktop preferences are saved on the machine running the worker, shared across projects for that account. They are not saved in browser storage. Reconnecting loads that host's preferences.

| Data | Default location |
| --- | --- |
| Appearance, default workflow, history limits, and debug preference | `~/.config/vops-runner/desktop-settings.json` |
| Submitted prompt text for recall and search | `~/.local/state/runner-desktop/prompt-history.sqlite3` |

Set `VOPS_RUNNER_CONFIG_DIR` to change the settings directory. Set `XDG_STATE_HOME` to an absolute path to change the prompt-history base directory; the database remains under its `runner-desktop/` subdirectory.

Prompt history keeps the latest 1,000 submitted text prompts in plaintext. It is separate from conversation transcripts and shared across workspaces using the same host account. The database has owner-only file permissions on POSIX systems. Drafts and image payloads are not copied into it. Avoid putting credentials or other secrets in prompts.

Runner manages conversation persistence, provider configuration, and credentials. See the [Runner documentation](https://github.com/vectorops/runner) for those settings.

## Troubleshooting

- **A provider cannot sign in:** check the terminal for details. Browser sign-in uses a provider-specific callback port; another login attempt can occupy that port.
- **A setting does not save:** check the displayed error and filesystem permissions on the settings directory. A failed save does not change the effective settings.
- **The connection drops:** the client reconnects and reloads the conversation state. A disconnected window does not mean the workflow stopped. Check the conversation before retrying an action whose outcome is unknown.

See [connection troubleshooting](connections.md#troubleshooting) for startup and authentication problems.
