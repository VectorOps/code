import type { RunnerApi } from "./runner-api";
import { object, type Params } from "../models/json";
import type { Session } from "../models/runner";

export type SessionList = {
  sessions: Session[];
  hasMore: boolean;
  loading: boolean;
};

export class SessionSync {
  private sessions = new Map<string, Session>();
  private values: Session[] = [];
  private updates = new Map<string, Session>();
  private generation = 0;
  private after: string | null = null;
  private loading = false;

  constructor(
    private api: RunnerApi,
    private publish: (view: SessionList) => void,
  ) {}

  private emit(changed = false) {
    if (changed) this.values = [...this.sessions.values()];
    this.publish({
      sessions: this.values,
      hasMore: this.after !== null,
      loading: this.loading,
    });
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
      const page = await this.api.sessions();
      if (!current()) return false;
      const refreshed = new Map(
        page.executions.map((session) => [session.execution_id, session]),
      );
      const retained = this.sessions.get(selected);
      if (retained && !refreshed.has(selected))
        refreshed.set(selected, retained);
      for (const [id, summary] of this.updates) refreshed.set(id, summary);
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

  async loadMore(active: () => boolean): Promise<boolean> {
    if (this.loading || this.after === null || !active()) return false;
    const generation = this.generation;
    const current = () => active() && generation === this.generation;
    const after = this.after;
    this.loading = true;
    this.updates.clear();
    this.emit();
    try {
      const page = await this.api.sessions(after);
      if (!current()) return false;
      for (const summary of page.executions) {
        if (!this.updates.has(summary.execution_id))
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

  apply(event: Params): { summary: Session; previous?: Session } | undefined {
    if (event.kind !== "execution.updated") return;
    const value = object(event.summary);
    if (
      typeof event.execution_id !== "string" ||
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
