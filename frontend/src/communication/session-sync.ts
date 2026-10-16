import type { RunnerApi } from "./runner-api";
import { object, type Params } from "../models/json";
import type { Session } from "../models/runner";

export class SessionSync {
  private sessions: Session[] = [];
  private updates = new Map<string, Session>();
  private generation = 0;

  constructor(
    private api: RunnerApi,
    private publish: (sessions: Session[]) => void,
  ) {}

  invalidate() {
    this.generation++;
    this.updates.clear();
  }

  async synchronize(active: () => boolean): Promise<boolean> {
    this.invalidate();
    const generation = this.generation;
    const current = () => active() && generation === this.generation;
    await this.api.subscribe();
    if (!current()) return false;
    const list = await this.api.sessions(current);
    if (!current()) return false;
    const refreshed = new Map(
      list.map((session) => [session.execution_id, session]),
    );
    for (const [id, summary] of this.updates) refreshed.set(id, summary);
    const ordered = this.sessions
      .filter((session) => refreshed.has(session.execution_id))
      .map((session) => refreshed.get(session.execution_id)!);
    const known = new Set(ordered.map((session) => session.execution_id));
    this.sessions = [
      ...ordered,
      ...[...refreshed.values()].filter(
        (session) => !known.has(session.execution_id),
      ),
    ];
    this.publish(this.sessions);
    return true;
  }

  get(executionId: string): Session | undefined {
    return this.sessions.find(
      (session) => session.execution_id === executionId,
    );
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
    this.updates.set(summary.execution_id, summary);
    this.sessions = previous
      ? this.sessions.map((session) =>
          session.execution_id === summary.execution_id ? summary : session,
        )
      : [summary, ...this.sessions];
    this.publish(this.sessions);
    return { summary, previous };
  }
}
