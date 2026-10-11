# Interface design guidelines

Design the application around sessions: conversation, persistent task state, and inspectable work. Users should be able to resume, redirect, and review work without reconstructing its state from the transcript.

This document defines intended outcomes and design decisions, not a claim that every feature is implemented. Expose an action only when its behavior is supported and its result can be confirmed.

## Contents

- [Core principles](#core-principles)
- [Layout and navigation](#layout-and-navigation)
- [History and continuity](#history-and-continuity)
- [Input and delivery](#input-and-delivery)
- [Context visibility](#context-visibility)
- [Activity and decisions](#activity-and-decisions)
- [Review and recovery](#review-and-recovery)
- [Status and lifecycle](#status-and-lifecycle)
- [Visual language and accessibility](#visual-language-and-accessibility)
- [Delivery priorities](#delivery-priorities)

## Core principles

- Keep the conversation central, but make current activity, pending decisions, and changed files visible outside the transcript.
- Preserve drafts, reading position, and review context when the user moves between views.
- Keep input usable while work is running. Drafting must not depend on whether a message can be delivered immediately.
- Distinguish requested actions from acknowledged actions and completed outcomes.
- Prefer predictable controls and progressive detail over more modes and customization.
- Never imply that stopping, disconnecting, reviewing, or closing a view undoes work.

A session represents one top-level workflow execution and its conversation. Changing the visible session changes the view, not the execution. Present readable titles; keep execution identifiers available as secondary details.

## Layout and navigation

Use three regions, with two visible by default, and an on-demand bottom drawer.

| Region | Purpose | Default |
| --- | --- | --- |
| Left navigation | Projects, recent and pinned sessions, search, attention filters | Visible and collapsible |
| Center conversation | Instructions, progress, decisions, responses, and results | Primary surface |
| Right work pane | Changed files, file content, review, plan, and preview | Closed; resizable and expandable |
| Bottom drawer | Terminal and detailed process output | Closed until requested |

Start with approximately 240 px for navigation. When review is open, allocate about 40-50% of the remaining width to it. Treat these as initial proportions to validate, not fixed constraints.

- Selecting a file or relevant activity opens its content in the work pane.
- Opening review preserves conversation position and the unsent draft.
- Expanded review retains a compact composer and session identity.
- Incoming activity never opens panels, changes the selected session, or steals focus.
- Narrow windows switch between conversation and review instead of compressing both into unreadable columns.
- Remember panel sizes, selected work tabs, and navigation visibility.
- Defer arbitrary docking and custom layouts until simpler arrangements prove insufficient.

## History and continuity

Separate returning to recent work from finding an older discussion.

| User goal | Design decision |
| --- | --- |
| Resume work | Group recent sessions by project |
| Find a discussion | Search titles and message content |
| Find relevant work | Filter by file path, repository, branch, or execution host |
| Monitor progress | Provide Running and Needs attention filters |
| Keep work accessible | Pin sessions |
| Reduce clutter | Archive without deleting history |
| Explore an alternative | Fork from a selected turn when supported |

Each session row shows a concise title, explicit status, and recency. Include project, branch, or host when needed to distinguish similar sessions. Avoid constantly reordering rows while the user navigates.

Search results include a matching excerpt and navigate to the corresponding message. Archived sessions remain searchable in the same interface; restoring them is a history action, not an application setting.

Switching sessions restores the draft, attachments, scroll position, and selected review file. Closing a session view does not stop its workflow. Keep closing, archiving, deleting, and stopping as distinct actions.

## Input and delivery

Make the effect of submission explicit. Do not silently reinterpret an instruction based on execution state.

| Action | Meaning | Presentation |
| --- | --- | --- |
| Send | Submit an instruction while idle | Primary composer action |
| Steer | Deliver a correction at the next supported execution boundary | Preferred primary action while running, when supported |
| Queue | Schedule an instruction after the current turn | Adjacent explicit option |
| Stop | Request cancellation of current work | Separate control with a stable position |

If steering or queueing is unavailable, retain the draft and explain when it can be sent. Do not display a nonfunctional substitute or pretend delivery has occurred.

- Show Waiting to be delivered until delivery is acknowledged.
- Display queued instructions separately from messages already received by the agent.
- Support editing, removing, and reordering queued instructions only before their delivery is committed.
- A submission with an unknown outcome after disconnect remains unresolved until reconciled; do not automatically send it again.
- Never replace Send with Stop under the pointer.
- Explain that cancellation can take time and that completed edits remain.

For example, a correction such as "Use PostgreSQL instead" is steering, while "After this, update the documentation" is a queued follow-up. Their displayed delivery state must make the distinction clear.

### Composer behavior

- Provide multiline editing and an expanded drafting surface.
- Preserve drafts and attachments per session across navigation and application restarts.
- Support pasted screenshots and dropped files with visible inclusion state.
- Represent explicit context as removable chips with previews.
- Offer file and selection references through autocomplete.
- Make commands discoverable through menus as well as keyboard entry.
- Make Enter-to-send configurable and show the alternative newline shortcut.
- Never submit during an active IME composition.
- Do not depend on double-Enter for an important action.
- Present ordinary conversation input as text, not a required structured-data editor.

## Context visibility

Make it clear what accompanies an instruction and what is merely available for inspection.

| Context category | Presentation |
| --- | --- |
| Attached evidence | Explicit chips for files, selections, screenshots, and logs |
| Available workspace | Project and execution environment the agent can access |
| Persistent instructions | Identifiable project rules and user preferences |

Selection chips show the filename and line range. Log attachments preview the included output. Attachment status distinguishes selected, preparing, included, and failed states where applicable.

Do not silently attach the active file. Any automatic context must be visible and removable. Paths belong to the execution host's project, not implicitly to the machine displaying the interface.

## Activity and decisions

Summarize routine progress; expand exceptions and pending decisions. Detailed records remain accessible without dominating the conversation.

| Activity | Default presentation |
| --- | --- |
| Reading and searching | Grouped summary with inspectable source details |
| File changes | Filenames and change counts linked to review |
| Running command | Concise command description, duration, and live state |
| Successful command | Collapsed result with exit status |
| Failed command | Error excerpt and recovery or retry state |
| Background process | Persistent entry with inspect and stop actions |
| Required decision | Expanded decision card |

Detailed inspection includes the exact action, arguments, working directory, execution host, output, timestamps, and exit status when available.

Preserve reading position during streaming. Follow the bottom only while the user is already following it; otherwise show a New activity indicator. Distinguish an individual operation failure from a failed session: recovery can continue after a failing command or test.

### Approval decisions

Keep intent, permissions, and isolation separate. Asking for implementation does not grant unrestricted permission, and granting permission does not establish isolation.

Each approval card answers:

| Question | Required information |
| --- | --- |
| What will happen? | Plain-language action description |
| What exactly will run or change? | Inspectable command or operation |
| Where? | Project path and execution host |
| Why is confirmation needed? | The applicable permission requirement |
| What does approval cover? | Exact action and duration or reuse scope |
| What can I choose? | Allow once, Deny, and supported additional scopes |

- Keep approvals in the conversation, with a persistent pending-decision indicator near the composer.
- Mark affected sessions Needs approval without blocking unrelated sessions.
- Name reusable grants precisely; never label a limited grant as approval for everything.
- Make saved permissions inspectable and revocable when supported.
- Offer Deny with instruction when the rejection can carry guidance.
- Reconcile decisions after reconnect and prevent duplicate or stale submissions across clients.
- Keep composer keyboard submission separate from approval controls. A newly arrived card never captures an intended message submission.

Permission to perform an action and review of its resulting edits are different decisions and must look different.

## Review and recovery

Offer explicit comparison scopes rather than one ambiguous Changes view.

| Scope | Meaning |
| --- | --- |
| Last turn | Changes attributable to the latest turn |
| This session | Changes reliably attributable to this session |
| Working tree | All current local changes, including unrelated user edits |
| Against base branch | Current differences from the selected base |

Only offer attribution-based scopes when ownership can be determined reliably. Do not imply all working-tree changes were produced by the agent.

Distinguish proposed edits from edits already on disk. Track Reviewed separately from either state. Avoid Accept when the underlying action is only marking existing edits as reviewed.

Allow line comments and collect them into one Send review feedback action so users can finish inspecting before requesting another pass.

### Recovery language

| Operation | Required meaning |
| --- | --- |
| Retry response | Start another attempt; explain whether existing changes remain |
| Fork conversation | Create an alternative conversational branch; do not imply file isolation |
| Revert turn changes | Reverse an identified set of edits |
| Restore checkpoint | Restore an explicitly defined snapshot and state scope |
| Stop execution | Request cancellation without undoing completed work |

Before restoring or reverting, show affected files, detect intervening edits, and explain conflicts. Never imply that restoring files reverses external effects such as publishing or database changes.

## Status and lifecycle

Show current state near the session title and in navigation, independently of connection health.

| Session state | Meaning |
| --- | --- |
| Running | Work is progressing |
| Needs approval | Permission is required to continue affected work |
| Needs answer | A user response is required |
| Ready for review | A result or change set is available for inspection |
| Completed | The workflow finished successfully |
| Failed | The workflow ended unsuccessfully |
| Stopped | Cancellation has been confirmed |

Show Stopping while cancellation is pending, rather than immediately claiming Stopped. A result can be ready for review while the workflow remains open for further conversation. Show simultaneous attention requirements as explicit indicators instead of hiding them behind one status.

- Display project, branch, and execution host in the session header where available.
- Treat disconnected state as stale knowledge, not proof that work stopped.
- Reconnect without resubmitting the last instruction.
- Restore acknowledged messages and authoritative pending decisions.
- Explain whether work continues when the application window closes.
- In in-process mode, closing the application stops its owned workflows; show a warning when work is active.
- In independent-worker mode, closing the application disconnects the view and leaves execution running.
- Notify for required decisions and completion, not every activity update.

### Setup and failure messages

Before starting work, make the selected project, execution target, model, and credential readiness understandable. Missing requirements should lead to a specific corrective action rather than a failure deep in the transcript.

Errors identify what failed, which session is affected, whether work continues, and what the user can do next. Keep technical details available without making raw records the primary explanation. Dismissing a notice does not imply the cause has been resolved.

## Visual language and accessibility

Use a restrained black-and-white default: white surfaces, near-black text, neutral borders, and subtle gray secondary surfaces. Reserve visual emphasis for the active session, primary action, pending decision, and current execution state.

- Use spacing, typography, borders, and explicit labels before adding decoration.
- Keep normal conversation easy to read; secondary metadata must not compete with message content.
- Use distinct type treatment for paths, commands, and exact output.
- Never communicate status, selection, or failure through color alone.
- Provide visible keyboard focus, descriptive control names, and a predictable navigation order.
- Make navigation, panel switching, resizing alternatives, and decision controls usable without a pointer.
- Preserve usability with text scaling, narrow windows, and reduced motion.
- Announce meaningful status changes without repeatedly interrupting assistive reading as output streams.
- Maintain contrast in disabled, selected, hover, and focus states.

## Delivery priorities

| Priority | Desired outcomes |
| --- | --- |
| Essential | Find and resume sessions; search history; retain drafts; understand execution state; stop work; inspect compact activity and approvals; review files and differences; recover after disconnect |
| Next | Redirect running work; queue follow-ups; collect review comments; inspect turn-level changes; safely revert; manage background processes; receive completion notifications |
| Later | Fork conversations; create isolated working copies; ask side questions; inspect previews; maintain substantial requirements and design artifacts; customize panel arrangements |

Keep everyday planning lightweight: show the current step near activity and the full plan in the work pane. Distinguish completed, active, blocked, and skipped steps. Show plan revisions without repeatedly copying the entire plan into the conversation. Larger work can introduce persistent requirements, design, and task artifacts later.

### Design review checklist

- Can the user identify the current project, session, host, and execution state without reading history?
- Can they switch sessions or open review without losing a draft or reading position?
- Is it clear whether an instruction is drafted, waiting, queued, or delivered?
- Can they distinguish an approval request from review of completed edits?
- Are permission scope and file-change ownership stated accurately?
- Do stop, close, retry, fork, and restore have visibly different effects?
- Does reconnect preserve work without silently duplicating an instruction?
- Can the primary workflows be completed using a keyboard and in a narrow window?

## Related

- [Project overview and current capabilities](../README.md)
- [Frontend and backend development loop](development/watch.md)