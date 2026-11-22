import type { RunnerApi } from "./runner-api";
import { object, type Params } from "../models/json";
import type { Session, SessionPage, SessionVisibility } from "../models/runner";

export type SessionList = {
  sessions: Session[];
  hasMore: boolean;
  loading: boolean;
};

export class SessionSync {
  private sessions = new Map<string, Session>();
  private values: Session[] = [];
  private updates = new Map<string, Session>();
  private deleted = new Set<string>();
  private generation = 0;
  private after: string | null = null;
  private loading = false;
  private hidden = new Set<string>();
  private visibilityRevision = -1;

  constructor(
    private api: RunnerApi,
    private publish: (view: SessionList) => void,
  ) {}

  private emit(changed = false) {
    if (changed)
      this.values = [...this.sessions.values()].map((session) =>
        this.hidden.has(session.execution_id)
          ? { ...session, hidden: true }
          : session,
      );
    this.publish({
      sessions: this.values,
      hasMore: this.after !== null,
      loading: this.loading,
    });
  }

  visibility(snapshot: SessionVisibility) {
    if (snapshot.revision <= this.visibilityRevision) return;
    this.visibilityRevision = snapshot.revision;
    this.hidden = new Set(snapshot.hidden_execution_ids);
    this.emit(true);
  }

  private async page(
    current: () => boolean,
    after?: string,
    includeHidden = false,
  ): Promise<SessionPage | undefined> {
    const executions = new Map<string, Session>();
    const cursors = new Set<string>();
    while (current()) {
      const page =
        after === undefined
          ? await this.api.sessions()
          : await this.api.sessions(after);
      if (!current()) return;
      if (page.visibility) this.visibility(page.visibility);
      for (const session of page.executions)
        executions.set(session.execution_id, session);
      const visible = [...executions.keys()].filter(
        (id) => !this.hidden.has(id),
      ).length;
      if (
        includeHidden ||
        visible >= 100 ||
        visible === executions.size ||
        page.next_after === null ||
        cursors.has(page.next_after)
      )
        return {
          executions: [...executions.values()],
          next_after: page.next_after,
        };
      after = page.next_after;
      cursors.add(after);
    }
  }

  invalidate() {
    this.generation++;
    this.updates.clear();
    this.after = null;
    this.loading = false;
    this.emit();
  }

  async synchronize(active: () => boolean, selected = ""): Promise<boolean> {
    this.invalidate();
    const generation = this.generation;
    const current = () => active() && generation === this.generation;
    this.loading = true;
    this.emit();
    try {
      await this.api.subscribe();
      if (!current()) return false;
      const page = await this.page(current);
      if (!page || !current()) return false;
      const refreshed = new Map(
        page.executions.map((session) => [session.execution_id, session]),
      );
      const retained = this.sessions.get(selected);
      if (retained && !refreshed.has(selected))
        refreshed.set(selected, retained);
      for (const [id, summary] of this.updates) refreshed.set(id, summary);
      for (const id of this.deleted) refreshed.delete(id);
      this.sessions = new Map([
        ...[...this.sessions.keys()]
          .filter((id) => refreshed.has(id))
          .map((id) => [id, refreshed.get(id)!] as const),
        ...refreshed,
      ]);
      this.after = page.next_after;
      this.emit(true);
      return true;
    } catch (reason) {
      if (!current()) return false;
      throw reason;
    } finally {
      if (current()) {
        this.loading = false;
        this.updates.clear();
        this.emit();
      }
    }
  }

  async loadMore(
    active: () => boolean,
    includeHidden = false,
  ): Promise<boolean> {
    if (this.loading || this.after === null || !active()) return false;
    const generation = this.generation;
    const current = () => active() && generation === this.generation;
    const after = this.after;
    this.loading = true;
    this.updates.clear();
    this.emit();
    try {
      const page = await this.page(current, after, includeHidden);
      if (!page || !current()) return false;
      for (const summary of page.executions) {
        if (
          !this.updates.has(summary.execution_id) &&
          !this.deleted.has(summary.execution_id)
        )
          this.sessions.set(summary.execution_id, summary);
      }
      this.after = page.next_after;
      this.emit(true);
      return true;
    } catch (reason) {
      if (!current()) return false;
      throw reason;
    } finally {
      if (current()) {
        this.loading = false;
        this.updates.clear();
        this.emit();
      }
    }
  }

  async refresh(executionId: string, active: () => boolean): Promise<void> {
    const generation = this.generation;
    const previous = this.get(executionId);
    const summary = await this.api.session(executionId);
    if (
      !active() ||
      this.deleted.has(executionId) ||
      generation !== this.generation ||
      this.get(executionId) !== previous
    )
      return;
    this.sessions.set(executionId, summary);
    if (this.loading) this.updates.set(executionId, summary);
    this.emit(true);
  }

  get(executionId: string): Session | undefined {
    return this.sessions.get(executionId);
  }

  remove(executionId: string) {
    this.deleted.add(executionId);
    this.updates.delete(executionId);
    if (this.sessions.delete(executionId)) this.emit(true);
  }

  apply(event: Params): { summary: Session; previous?: Session } | undefined {
    if (event.kind !== "execution.updated") return;
    const value = object(event.summary);
    if (
      typeof event.execution_id !== "string" ||
      this.deleted.has(event.execution_id) ||
      value.execution_id !== event.execution_id ||
      typeof value.status !== "string" ||
      typeof value.active !== "boolean" ||
      typeof value.pending_inputs !== "number"
    )
      return;
    const summary = value as Session;
    const previous = this.get(summary.execution_id);
    if (this.loading) this.updates.set(summary.execution_id, summary);
    if (previous) this.sessions.set(summary.execution_id, summary);
    else
      this.sessions = new Map([
        [summary.execution_id, summary],
        ...this.sessions,
      ]);
    this.emit(true);
    return { summary, previous };
  }
}
