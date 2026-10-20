import type { RunnerApi } from "./runner-api";
import { object, text, type Params } from "../models/json";
import type { HistoryItem, HistoryPage } from "../models/runner";
import * as policy from "../models/history-limits";

export type HistoryWindow = {
  items: HistoryItem[];
  hasOlder: boolean;
  loadingOlder: boolean;
};

export function emptyHistory(): HistoryWindow {
  return { items: [], hasOlder: false, loadingOlder: false };
}

export class HistorySync {
  private page = emptyHistory();
  private byId = new Map<string, HistoryItem>();
  private generation = 0;
  private floor = 0;
  private reviewing = false;
  private trimRequested = false;
  private limits: policy.HistoryLimits;

  constructor(
    private api: RunnerApi,
    private executionId: string,
    limits = policy.defaultHistoryLimits,
  ) {
    this.limits = policy.historyLimits(limits);
  }

  get snapshot(): HistoryWindow {
    return this.page;
  }

  configure(limits: policy.HistoryLimits) {
    this.limits = policy.historyLimits(limits);
  }

  invalidate() {
    this.generation++;
    if (this.page.loadingOlder)
      this.page = { ...this.page, loadingOlder: false };
  }

  follow(following: boolean) {
    this.reviewing = !following;
  }

  trim(beforePosition: number): HistoryWindow {
    if (this.reviewing || this.page.loadingOlder) return this.page;
    const items = this.page.items;
    if (items.length > this.limits.maximum) this.trimRequested = true;
    if (!this.trimRequested) return this.page;
    let count = 0;
    while (
      count < items.length - this.limits.target &&
      items[count].position < beforePosition
    )
      count++;
    if (count) {
      this.floor = Math.max(this.floor, items[count - 1].position);
      for (const item of items.slice(0, count))
        this.byId.delete(text(object(item.step).id));
      this.page = { ...this.page, items: items.slice(count), hasOlder: true };
    }
    if (this.page.items.length <= this.limits.target)
      this.trimRequested = false;
    return this.page;
  }

  apply(event: Params): HistoryWindow | undefined {
    if (
      event.execution_id !== this.executionId ||
      ![
        "step.created",
        "step.updated",
        "step.completed",
        "step.failed",
      ].includes(text(event.kind)) ||
      typeof event.position !== "number" ||
      !Number.isInteger(event.position) ||
      event.position < 1 ||
      !text(object(event.step).id)
    )
      return;
    const step = object(event.step);
    if (
      !this.byId.has(text(step.id)) &&
      (event.position <= this.floor || Number(step.revision ?? 1) !== 1)
    )
      return this.page;
    return this.merge([{ position: event.position, step: event.step }]);
  }

  async refresh(active: () => boolean): Promise<HistoryWindow> {
    this.invalidate();
    const generation = this.generation;
    const limit = this.limits.initial;
    const updates: HistoryPage = { items: [], next_cursor: null };
    do {
      const page = await this.api.history(this.executionId, {
        limit: Math.min(500, limit - updates.items.length),
        ...(updates.next_cursor ? { cursor: updates.next_cursor } : {}),
      });
      if (!active() || generation !== this.generation) return this.page;
      updates.items.unshift(...page.items);
      updates.next_cursor = page.next_cursor;
      if (!page.items.length) break;
    } while (updates.next_cursor && updates.items.length < limit);
    if (!this.reviewing) {
      const last = updates.items.at(-1)?.position ?? 0;
      const ids = new Set(
        updates.items.map((item) => text(object(item.step).id)),
      );
      const retained = this.page.items.filter(
        (item) => item.position > last || ids.has(text(object(item.step).id)),
      );
      this.byId = new Map(
        retained.map((item) => [text(object(item.step).id), item]),
      );
      if (
        retained.length !== this.page.items.length ||
        this.page.hasOlder !== Boolean(updates.next_cursor)
      )
        this.page = {
          ...this.page,
          items: retained,
          hasOlder: Boolean(updates.next_cursor),
        };
      this.floor = Math.max(this.floor, (updates.items[0]?.position ?? 1) - 1);
    } else if (!this.page.items.length) {
      this.page = { ...this.page, hasOlder: Boolean(updates.next_cursor) };
    }
    return this.merge(updates.items);
  }

  async loadOlder(active: () => boolean): Promise<HistoryWindow> {
    const first = this.page.items[0];
    if (!first || !this.page.hasOlder || this.page.loadingOlder || !active())
      return this.page;
    this.reviewing = true;
    this.trimRequested = true;
    const generation = this.generation;
    this.page = { ...this.page, loadingOlder: true };
    try {
      const older: HistoryPage = await this.api.history(this.executionId, {
        limit: this.limits.older,
        before_step_id: text(object(first.step).id),
      });
      if (active() && generation === this.generation) {
        this.merge(older.items);
        this.page = { ...this.page, hasOlder: Boolean(older.next_cursor) };
      }
    } finally {
      if (generation === this.generation)
        this.page = { ...this.page, loadingOlder: false };
    }
    return this.page;
  }

  private merge(items: HistoryItem[]): HistoryWindow {
    let ordered: HistoryItem[] | undefined;
    for (const item of items) {
      const step = object(item.step);
      const id = text(step.id);
      if (!id) throw new Error("History step is missing its persistent ID");
      const previous = this.byId.get(id);
      if (
        previous &&
        Number(step.revision ?? 1) <=
          Number(object(previous.step).revision ?? 1)
      )
        continue;
      this.byId.set(id, item);
      ordered ??= [...this.page.items];
      let low = 0;
      let high = ordered.length;
      while (low < high) {
        const middle = (low + high) >>> 1;
        if (ordered[middle].position < item.position) low = middle + 1;
        else high = middle;
      }
      if (previous) ordered[low] = item;
      else ordered.splice(low, 0, item);
    }
    if (ordered) this.page = { ...this.page, items: ordered };
    return this.page;
  }
}
