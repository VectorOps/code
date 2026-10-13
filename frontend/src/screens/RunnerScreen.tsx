import { useState } from "react";
import type { ConnectionSettings } from "../models/runner";
import { useRunner } from "../hooks/useRunner";
import { ConnectionForm } from "../components/ConnectionForm";
import { NewSessionForm } from "../components/NewSessionForm";
import { Workspace } from "./Workspace";
import { Settings } from "./Settings";

export function RunnerScreen({ initial }: { initial: ConnectionSettings }) {
  const runner = useRunner(initial);
  const [connection, setConnection] = useState(initial);
  const [draft, setDraft] = useState({ reference: "", input: "{}" });
  return (
    <Workspace
      settings={
        <Settings
          api={runner.api}
          connected={runner.status === "Connected"}
          connection={runner.settings}
        />
      }
      sessions={runner.sessions}
      workflows={runner.workflows}
      selected={runner.selected}
      onSelect={runner.select}
      history={runner.history.items}
      execution={runner.execution}
      branching={runner.branching?.execution_id === runner.selected}
      hasOlder={Boolean(runner.history.next_cursor)}
      pending={runner.pending}
      status={runner.status}
      endpoint={runner.settings.endpoint}
      enabled={runner.enabled}
      busy={runner.busy}
      error={runner.error}
      dismiss={runner.dismiss}
      onStop={async () => {
        await runner.stop();
      }}
      onResume={async () => {
        await runner.resume();
      }}
      onReply={runner.reply}
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
          start={runner.start}
        />
      }
    />
  );
}
