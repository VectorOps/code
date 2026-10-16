import type { RunnerApi } from "./runner-api";
import { object, text, type Params } from "../models/json";
import type { HistoryItem, HistoryPage } from "../models/runner";

export class HistorySync {
  private page: HistoryPage = { items: [], next_cursor: null };
  private byId = new Map<string, HistoryItem>();

  constructor(
    private api: RunnerApi,
    private executionId: string,
  ) {}

  get snapshot(): HistoryPage {
    return this.page;
  }

  apply(event: Params): HistoryPage | undefined {
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
    return this.merge([{ position: event.position, step: event.step }]);
  }

  async refresh(active: () => boolean): Promise<HistoryPage> {
    const updates = await this.api.history(
      this.executionId,
      0,
      undefined,
      active,
    );
    return active() ? this.merge(updates.items) : this.page;
  }

  private merge(items: HistoryItem[]): HistoryPage {
    let changed = false;
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
      changed = true;
    }
    if (changed)
      this.page = {
        items: [...this.byId.values()].sort((a, b) => a.position - b.position),
        next_cursor: null,
      };
    return this.page;
  }
}
