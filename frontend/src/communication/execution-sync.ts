import { HistorySync } from "./history-sync";
import type { RunnerApi } from "./runner-api";
import { object, text, type Params } from "../models/json";
import type * as models from "../models/runner";

export type ExecutionView = {
  execution?: models.Execution;
  history: models.HistoryPage;
  pending: models.InputRequest[];
  steering?: {
    address: models.InboxAddress;
    messages: models.InboxMessage[];
    count: number;
  };
};

type Inbox = models.InboxStatus & { messages: models.InboxMessage[] };

export class ExecutionSync {
  private history: HistorySync;
  private state: ExecutionView = {
    history: { items: [], next_cursor: null },
    pending: [],
  };
  private inboxes = new Map<string, Inbox>();
  private buffered: Params[] = [];
  private loading = true;
  private generation = 0;

  constructor(
    private api: RunnerApi,
    private executionId: string,
    private publish: (view: ExecutionView) => void,
  ) {
    this.history = new HistorySync(api, executionId);
  }

  invalidate() {
    this.generation++;
    this.loading = true;
    this.buffered = [];
  }

  async synchronize(active: () => boolean) {
    this.loading = true;
    const generation = ++this.generation;
    const current = () => active() && generation === this.generation;
    const execution = await this.api.execution(this.executionId);
    if (!current()) return;
    const [, pending, inboxes] = await Promise.all([
      this.history.refresh(current),
      this.api.pending(this.executionId, current),
      ["running", "paused", "starting"].includes(execution.status)
        ? this.api.inboxes(this.executionId).catch(() => [])
        : Promise.resolve([]),
    ]);
    if (!current()) return;
    const mailbox = inboxes.find(
      (inbox) =>
        inbox.address.workflow_id === execution.root_workflow_id &&
        inbox.address.conversation_id === "steer" &&
        inbox.status === "open",
    );
    const messages = mailbox
      ? await this.api.inboxMessages(mailbox.address)
      : [];
    if (!current()) return;
    this.state = { execution, history: this.history.snapshot, pending };
    this.inboxes.clear();
    if (mailbox)
      this.inboxes.set(mailbox.address.workflow_id, { ...mailbox, messages });
    for (const event of this.buffered) this.update(event);
    this.buffered = [];
    this.loading = false;
    this.emit();
  }

  apply(event: Params) {
    if (event.execution_id !== this.executionId) return;
    const history = this.history.apply(event);
    if (history) {
      if (history !== this.state.history) {
        this.state = { ...this.state, history };
        this.emit();
      }
    } else if (this.update(event)) {
      if (this.loading) this.buffered.push(event);
      this.emit();
    }
  }

  private update(event: Params): boolean {
    const execution = object(event.execution);
    const request = object(event.request);
    const address = object(event.address);
    if (
      event.kind === "execution.state_updated" &&
      execution.id === this.executionId
    ) {
      this.state = { ...this.state, execution: execution as models.Execution };
    } else if (event.kind === "execution.updated") {
      const summary = object(event.summary);
      if (summary.execution_id !== this.executionId) return false;
      if (this.state.execution && typeof summary.status === "string")
        this.state = {
          ...this.state,
          execution: { ...this.state.execution, status: summary.status },
        };
    } else if (
      (event.kind === "mailbox.requested" ||
        event.kind === "mailbox.answered") &&
      request.execution_id === this.executionId &&
      typeof request.request_id === "string"
    ) {
      const pending = this.state.pending.filter(
        (entry) => entry.request_id !== request.request_id,
      );
      if (event.kind === "mailbox.requested" && request.status === "pending")
        pending.push(request as models.InputRequest);
      this.state = { ...this.state, pending };
    } else if (event.kind === "mailbox.cancelled") {
      const ids = event.request_ids;
      if (!Array.isArray(ids)) return false;
      this.state = {
        ...this.state,
        pending: this.state.pending.filter(
          (entry) => !ids.includes(entry.request_id),
        ),
      };
    } else if (
      ["mailbox.state_updated", "mailbox.enqueued", "mailbox.removed"].includes(
        text(event.kind),
      ) &&
      address.execution_id === this.executionId &&
      address.conversation_id === "steer" &&
      typeof address.workflow_id === "string" &&
      (event.status === "open" || event.status === "closed") &&
      typeof event.queued_messages === "number"
    ) {
      const previous = this.inboxes.get(address.workflow_id);
      let messages = previous?.messages ?? [];
      if (event.kind === "mailbox.enqueued") {
        const message = object(event.message);
        if (typeof message.id !== "string") return false;
        messages = [
          ...messages.filter((entry) => entry.id !== message.id),
          message as models.InboxMessage,
        ];
      } else if (event.kind === "mailbox.removed")
        messages = messages.filter((entry) => entry.id !== event.message_id);
      if (event.queued_messages === 0) messages = [];
      this.inboxes.set(address.workflow_id, {
        address: address as models.InboxAddress,
        status: event.status,
        queued_messages: event.queued_messages,
        messages,
      });
    } else return false;
    return true;
  }

  private emit() {
    const mailbox = this.inboxes.get(
      this.state.execution?.root_workflow_id ?? "",
    );
    this.publish({
      ...this.state,
      steering:
        mailbox?.status === "open"
          ? {
              address: mailbox.address,
              count: mailbox.queued_messages,
              messages: mailbox.messages,
            }
          : undefined,
    });
  }
}
