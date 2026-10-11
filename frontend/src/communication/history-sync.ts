import type { RunnerApi } from "./runner-api";
import { object, text } from "../models/json";
import type { HistoryItem, HistoryPage } from "../models/runner";

export class HistorySync {
  private page: HistoryPage = { items: [], next_cursor: null };
  private byId = new Map<string, HistoryItem>();
  private dirty = new Set<string>();
  private unfinished = new Set<string>();
  private position = 0;

  constructor(
    private api: RunnerApi,
    private executionId: string,
  ) {}

  invalidate(id: string) {
    if (id) this.dirty.add(id);
  }

  reconcile() {
    for (const id of this.byId.keys()) this.dirty.add(id);
  }

  async refresh(active: () => boolean): Promise<HistoryPage> {
    const ids = new Set([...this.unfinished, ...this.dirty]);
    this.dirty.clear();
    try {
      const updates = await this.api.history(
        this.executionId,
        this.position,
        undefined,
        active,
      );
      const targeted = [...ids];
      for (let start = 0; start < targeted.length && active(); start += 500) {
        const page = await this.api.history(
          this.executionId,
          0,
          targeted.slice(start, start + 500),
          active,
        );
        updates.items.push(...page.items);
      }
      if (!active()) return this.page;
      let changed = false;
      for (const item of updates.items) {
        const step = object(item.step);
        const id = text(step.id);
        if (!id) throw new Error("History step is missing its persistent ID");
        const previous = this.byId.get(id);
        if (
          !previous ||
          previous.position !== item.position ||
          JSON.stringify(previous.step) !== JSON.stringify(item.step)
        ) {
          this.byId.set(id, item);
          changed = true;
        }
        const state = text(object(step.outcome).kind);
        if (["completed", "failed", "cancelled"].includes(state))
          this.unfinished.delete(id);
        else this.unfinished.add(id);
        this.position = Math.max(this.position, item.position);
      }
      if (changed)
        this.page = {
          items: [...this.byId.values()].sort(
            (a, b) => a.position - b.position,
          ),
          next_cursor: null,
        };
      return this.page;
    } catch (reason) {
      for (const id of ids) this.dirty.add(id);
      throw reason;
    }
  }
}
