import { useMemo, useRef, useState } from "react";
import { PromptHistoryApi } from "../communication/prompt-history";
import { FileSearchApi } from "../communication/file-search";
import type { ConnectionSettings } from "../models/runner";
import { useRunner } from "../hooks/useRunner";
import { useWorkspaceTitle } from "../hooks/useWorkspaceTitle";
import { ConnectionForm } from "../components/ConnectionForm";
import {
  NewSessionForm,
  type NewSessionDraft,
} from "../components/NewSessionForm";
import { Workspace } from "./Workspace";
import { Settings } from "./Settings";
import { ToolPermissions } from "./ToolPermissions";
import {
  resolveDefaultWorkflow,
  useDefaultWorkflow,
} from "../hooks/useDefaultWorkflow";
import type { Params } from "../models/json";
import { workflowDefaults } from "../models/workflow-input";

export function RunnerScreen({ initial }: { initial: ConnectionSettings }) {
  const runner = useRunner(initial);
  useWorkspaceTitle(runner.api, runner.status === "Connected");
  const promptHistory = useMemo(
    () =>
      runner.api ? new PromptHistoryApi(runner.api.connection) : undefined,
    [runner.api],
  );
  const [connection, setConnection] = useState(initial);
  const fileSearch = useMemo(
    () =>
      runner.api && runner.status === "Connected"
        ? new FileSearchApi(runner.api.connection)
        : undefined,
    [runner.api, runner.status],
  );
  const defaultWorkflow = useDefaultWorkflow();
  const [launchError, setLaunchError] = useState("");
  const starting = useRef(false);
  const [draft, setDraft] = useState<NewSessionDraft>({
    reference: "",
    input: "{}",
  });
  async function start(reference: string, input: Params) {
    if (!runner.enabled || starting.current) return false;
    starting.current = true;
    setLaunchError("");
    try {
      return await runner.start(reference, input);
    } finally {
      starting.current = false;
    }
  }
  async function create() {
    const workflow = resolveDefaultWorkflow(
      runner.workflows,
      defaultWorkflow.reference,
    );
    if (!workflow) {
      setLaunchError(
        `Default workflow “${defaultWorkflow.reference}” is unavailable. Choose another in Settings or use Advanced.`,
      );
      return false;
    }
    return start(
      workflow.descriptor.reference,
      workflowDefaults(workflow.descriptor.input_schema),
    );
  }
  return (
    <Workspace
      onCreate={create}
      promptHistory={promptHistory}
      fileSearch={fileSearch}
      settings={
        <Settings
          api={runner.api}
          connected={runner.status === "Connected"}
          connection={runner.settings}
          workflows={runner.workflows}
          defaultWorkflow={defaultWorkflow.reference}
          onDefaultWorkflowChange={defaultWorkflow.change}
        />
      }
      workflowPermissions={
        runner.api && runner.selected && runner.execution?.root_workflow_id ? (
          <ToolPermissions
            key={`${runner.selected}/${runner.execution.root_workflow_id}`}
            api={runner.api}
            scope={{
              scope: "workflow",
              execution_id: runner.selected,
              workflow_id: runner.execution.root_workflow_id,
            }}
          />
        ) : undefined
      }
      sessions={runner.sessions}
      hasMoreSessions={runner.hasMoreSessions}
      sessionsLoading={runner.sessionsLoading}
      onLoadMoreSessions={runner.loadMoreSessions}
      activitySessions={runner.activitySessions}
      workflows={runner.workflows}
      selected={runner.selected}
      onSelect={runner.select}
      history={runner.history.items}
      execution={runner.execution}
      branching={runner.branching?.execution_id === runner.selected}
      hasOlder={Boolean(runner.history.next_cursor)}
      pending={runner.pending}
      steering={runner.steering}
      status={runner.status}
      endpoint={runner.settings.endpoint}
      enabled={runner.enabled}
      busy={runner.busy}
      error={launchError || runner.error}
      dismiss={() => {
        setLaunchError("");
        runner.dismiss();
      }}
      onStop={async () => {
        await runner.stop();
      }}
      onResume={async () => {
        await runner.resume();
      }}
      onReply={runner.reply}
      onSteer={runner.steer}
      onBranch={runner.branch}
      connection={
        <ConnectionForm
          value={connection}
          change={setConnection}
          connect={() => runner.connect(connection)}
        />
      }
      newSession={
        <NewSessionForm
          workflows={runner.workflows}
          enabled={runner.enabled}
          value={draft}
          change={setDraft}
          start={start}
        />
      }
    />
  );
}
