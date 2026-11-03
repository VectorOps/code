import * as React from "react";
import { AppearanceControl } from "../components/Appearance";
import type { RunnerApi } from "../communication/runner-api";
import type { ConnectionSettings, Workflow } from "../models/runner";
import { Authentication } from "./Authentication";
import { LlmProfiles } from "./LlmProfiles";
import { WorkflowSettings } from "./WorkflowSettings";
import { ToolPermissions } from "./ToolPermissions";
import { WorktreeManager } from "./WorktreeManager";
import { resolveDefaultWorkflow } from "../hooks/useDefaultWorkflow";
import "./settings.css";
import { HistorySettings } from "../components/HistorySettings";
import {
  defaultHistoryLimits,
  type HistoryLimits,
} from "../models/history-limits";

const categories = {
  appearance: "Visual layout",
  conversations: "Conversations",
  authentication: "LLM authentication",
  profiles: "LLM profiles",
  permissions: "Tool permissions",
  workflows: "Workflows",
  worktrees: "Worktrees",
};

type Category = keyof typeof categories;

export function Settings({
  api,
  connected,
  connection,
  workflows = [],
  defaultWorkflow = "agent",
  onDefaultWorkflowChange,
  historyLimits = defaultHistoryLimits,
  onHistoryLimitsChange,
}: {
  api?: RunnerApi;
  connected: boolean;
  connection: ConnectionSettings;
  workflows?: Workflow[];
  defaultWorkflow?: string;
  onDefaultWorkflowChange?: (reference: string) => void;
  historyLimits?: HistoryLimits;
  onHistoryLimitsChange?: (limits: HistoryLimits) => void;
}) {
  const [category, setCategory] = React.useState<Category>("appearance");
  const title = categories[category];
  return (
    <div className="settings-page">
      <nav aria-label="Settings categories" className="settings-nav">
        {(Object.keys(categories) as Category[]).map((value) => (
          <button
            key={value}
            aria-current={category === value ? "page" : undefined}
            onClick={() => setCategory(value)}
          >
            {categories[value]}
          </button>
        ))}
      </nav>
      <section className="settings-content" aria-label={title}>
        <h3>{title}</h3>
        {category === "appearance" ? (
          <AppearanceControl />
        ) : category === "conversations" ? (
          <>
            <label>
              Default workflow
              <select
                value={defaultWorkflow}
                disabled={!connected || !onDefaultWorkflowChange}
                onChange={(event) =>
                  onDefaultWorkflowChange?.(event.target.value)
                }
              >
                <option value="agent">agent (default)</option>
                {defaultWorkflow !== "agent" &&
                  !workflows.some(
                    (item) => item.descriptor.reference === defaultWorkflow,
                  ) && (
                    <option value={defaultWorkflow}>
                      {defaultWorkflow} (unavailable)
                    </option>
                  )}
                {workflows
                  .filter((item) => item.descriptor.reference !== "agent")
                  .map((item) => (
                    <option
                      key={item.descriptor.reference}
                      value={item.descriptor.reference}
                    >
                      {item.descriptor.name}
                    </option>
                  ))}
              </select>
            </label>
            <p>
              New conversation and Cmd/Ctrl+N start this workflow. Saved on this
              runner host. Use Advanced or Cmd/Ctrl+Shift+N to choose a workflow
              and its inputs.
            </p>
            {!connected ? (
              <p role="status">Connect to a runner to choose a workflow.</p>
            ) : (
              !resolveDefaultWorkflow(workflows, defaultWorkflow) && (
                <p role="status">
                  This default workflow is not available on the connected
                  runner.
                </p>
              )
            )}
            {onHistoryLimitsChange && (
              <HistorySettings
                limits={historyLimits}
                change={onHistoryLimitsChange}
              />
            )}
          </>
        ) : !connected || !api ? (
          <p role="status">Runner disconnected.</p>
        ) : category === "worktrees" ? (
          <WorktreeManager key={`${connection.endpoint}/${connection.token}`} />
        ) : category === "workflows" ? (
          <WorkflowSettings
            key={`${connection.endpoint}/${connection.token}`}
            api={api}
          />
        ) : category === "permissions" ? (
          <ToolPermissions
            key={`${connection.endpoint}/${connection.token}`}
            api={api}
          />
        ) : category === "profiles" ? (
          <LlmProfiles
            key={`${connection.endpoint}/${connection.token}`}
            api={api}
          />
        ) : (
          <Authentication
            key={`${connection.endpoint}/${connection.token}`}
            api={api}
            connection={connection}
          />
        )}
      </section>
    </div>
  );
}
