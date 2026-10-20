import type { RunnerConnection } from "./websocket";
import type { Worktree, WorktreeCreate } from "../models/worktrees";

export class WorktreesApi {
  constructor(private readonly connection: RunnerConnection) {}

  async list(refresh = false): Promise<Worktree[]> {
    const result: Worktree[] = [];
    let after: string | null = null;
    do {
      const page: { worktrees: Worktree[]; next_after: string | null } =
        await this.connection.request("worktrees.list", {
          after,
          limit: 100,
          include_deleted: true,
          refresh,
        });
      result.push(...page.worktrees);
      after = page.next_after;
    } while (after);
    return result;
  }

  get(worktree_id: string) {
    return this.connection.request<Worktree>("worktrees.get", { worktree_id });
  }

  create(input: WorktreeCreate, idempotency_key: string) {
    return this.connection.request<Worktree>("worktrees.create", {
      ...input,
      idempotency_key,
    });
  }

  update(worktree_id: string, name: string, description: string) {
    return this.connection.request<Worktree>("worktrees.update", {
      worktree_id,
      name,
      description,
    });
  }

  delete(worktree_id: string, force = false) {
    return this.connection.request<Worktree>("worktrees.delete", {
      worktree_id,
      force,
    });
  }
}
