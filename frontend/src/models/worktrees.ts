export type Worktree = {
  id: string;
  name: string;
  description: string;
  path: string;
  repository: { root: string; identity: string };
  branch: string;
  base_commit: string;
  origin: {
    kind: "user" | "workflow";
    execution_id?: string | null;
    workflow_id?: string | null;
    step_id?: string | null;
  };
  status: "creating" | "ready" | "deleting" | "deleted" | "failed" | "missing";
  created_at: string;
  updated_at: string;
  last_used_at?: string | null;
  inspection?: {
    head: string;
    branch?: string | null;
    dirty: boolean;
    inspected_at: string;
  } | null;
  error?: string | null;
};

export type WorktreeCreate = {
  name: string;
  description: string;
  branch?: string;
  base_ref: string;
};
