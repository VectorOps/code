import * as React from "react";
import {
  DesktopSettingsProvider,
  useDesktopSettings,
} from "../components/DesktopSettings";
import { AppearanceProvider } from "../components/Appearance";
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
import { WorktreesProvider } from "../components/Worktrees";
import { useHistoryLimits } from "../hooks/useHistoryLimits";

export function RunnerScreen({ initial }: { initial: ConnectionSettings }) {
  return (
    <DesktopSettingsProvider>
      <AppearanceProvider>
        <ConnectedRunner initial={initial} />
      </AppearanceProvider>
    </DesktopSettingsProvider>
  );
}

function ConnectedRunner({ initial }: { initial: ConnectionSettings }) {
  const preferences = useDesktopSettings();
  const historySettings = useHistoryLimits();
  const runner = useRunner(initial, historySettings.limits);
  const settingsConnection =
    runner.status === "Connected" ? runner.api?.connection : undefined;
  React.useLayoutEffect(
    () => preferences.attach(settingsConnection),
    [preferences.attach, settingsConnection],
  );
  useWorkspaceTitle(runner.api, runner.status === "Connected");
  const promptHistory = React.useMemo(
    () =>
      runner.api ? new PromptHistoryApi(runner.api.connection) : undefined,
    [runner.api],
  );
  const [connection, setConnection] = React.useState(initial);
  const fileSearch = React.useMemo(
    () =>
      runner.api && runner.status === "Connected"
        ? new FileSearchApi(runner.api.connection)
        : undefined,
    [runner.api, runner.status],
  );
  const defaultWorkflow = useDefaultWorkflow();
  const [launchError, setLaunchError] = React.useState("");
  const starting = React.useRef(false);
  const [draft, setDraft] = React.useState<NewSessionDraft>({
    reference: "",
    input: "{}",
  });
  async function start(reference: string, input: Params, worktreeId?: string) {
    if (!runner.enabled || starting.current) return false;
    starting.current = true;
    setLaunchError("");
    try {
      return worktreeId
        ? await runner.start(reference, input, worktreeId)
        : await runner.start(reference, input);
    } finally {
      starting.current = false;
    }
  }
  async function create() {
    if (!preferences.ready) return false;
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
    <WorktreesProvider
      runner={runner.status === "Connected" ? runner.api : undefined}
    >
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
            onDefaultWorkflowChange={
              preferences.ready ? defaultWorkflow.change : undefined
            }
            historyLimits={historySettings.limits}
            onHistoryLimitsChange={
              preferences.ready ? historySettings.change : undefined
            }
          />
        }
        workflowPermissions={
          runner.api &&
          runner.selected &&
          runner.execution?.root_workflow_id ? (
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
        workflowStates={runner.workflowStates}
        branching={runner.branching?.execution_id === runner.selected}
        hasOlder={runner.history.hasOlder}
        historyLimits={historySettings.limits}
        historyLoading={runner.history.loadingOlder}
        onLoadOlder={runner.loadOlderHistory}
        onHistoryFollowing={runner.followHistory}
        onHistoryTrim={runner.trimHistory}
        pending={runner.pending}
        steering={runner.steering}
        status={runner.status}
        endpoint={runner.settings.endpoint}
        managedConnection={runner.settings.managed === true}
        enabled={runner.enabled}
        busy={runner.busy}
        error={launchError || runner.error || preferences.error}
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
        onDelete={runner.deleteExecution}
        onReply={runner.reply}
        onSteer={runner.steer}
        onRemoveSteering={runner.removeSteering}
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
    </WorktreesProvider>
  );
}
