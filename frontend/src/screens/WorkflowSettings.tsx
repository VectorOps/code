import { DebugOnly } from "../components/Diagnostics";
import * as React from "react";
import { ConfigurationApi } from "../communication/configuration-api";
import type { ConfigurationSnapshot } from "../communication/configuration-api";
import type { RunnerApi } from "../communication/runner-api";
import * as controls from "../components/SettingsControls";
import { WorkflowSelect } from "../components/WorkflowSelect";
import { SettingsIcon } from "../components/SettingsIcon";
import { object } from "../models/json";
import type { Json } from "../models/json";
import type { Workflow } from "../models/runner";

const prompts: controls.SettingDefinition[] = [
  {
    name: "system_prompt",
    label: "System prompt",
    multiline: true,
    allowEmpty: true,
    description:
      "Replace the built-in base prompt. Reset to use the built-in prompt.",
  },
  {
    name: "system_append",
    label: "Additional instructions",
    multiline: true,
    allowEmpty: true,
    description: "Append instructions to the base prompt.",
  },
];
const groups: {
  name: string;
  label: string;
  fields: controls.SettingDefinition[];
}[] = [
  {
    name: "agent",
    label: "Agent",
    fields: [
      {
        name: "tool_collection",
        label: "Agent tool collection",
        pattern: "[A-Za-z_][A-Za-z0-9_-]*",
        placeholder: "agent",
      },
      ...prompts,
    ],
  },
  {
    name: "helper",
    label: "Helper",
    fields: [
      {
        name: "thinking_level",
        label: "Helper thinking level",
        placeholder: "Inherit shared defaults",
      },
      {
        name: "temperature",
        label: "Helper temperature",
        numeric: true,
        min: 0,
        max: 2,
        step: "any",
      },
      {
        name: "max_rounds",
        label: "Maximum rounds",
        numeric: true,
        min: 1,
        step: 1,
        placeholder: "30",
      },
      {
        name: "tool_collection",
        label: "Helper tool collection",
        pattern: "[A-Za-z_][A-Za-z0-9_-]*",
        placeholder: "helper",
      },
      ...prompts,
    ],
  },
];

function message(reason: unknown) {
  return reason instanceof Error ? reason.message : "Request failed.";
}

export function WorkflowSettings({ api: runner }: { api: RunnerApi }) {
  const api = React.useMemo(
    () => new ConfigurationApi(runner.connection),
    [runner],
  );
  const [snapshot, setSnapshot] = React.useState<ConfigurationSnapshot>();
  const [workflows, setWorkflows] = React.useState<Workflow[]>([]);
  const [discoveryError, setDiscoveryError] = React.useState("");
  const [discovering, setDiscovering] = React.useState(false);
  const [busy, setBusy] = React.useState(true);
  const [error, setError] = React.useState("");
  const [notice, setNotice] = React.useState("");
  const generation = React.useRef(0);
  const discoveryGeneration = React.useRef(0);

  async function discover() {
    const current = ++discoveryGeneration.current;
    setDiscovering(true);
    setDiscoveryError("");
    try {
      const next = await runner.workflows(
        () => current === discoveryGeneration.current,
      );
      if (current === discoveryGeneration.current) setWorkflows(next);
    } catch (reason) {
      if (current === discoveryGeneration.current)
        setDiscoveryError(
          `Workflow discovery unavailable. Existing selections are preserved. ${message(reason)}`,
        );
    } finally {
      if (current === discoveryGeneration.current) setDiscovering(false);
    }
  }

  async function load(operation?: () => Promise<unknown>, success = "") {
    const current = ++generation.current;
    let applied = false;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      if (operation) {
        await operation();
        applied = true;
      }
      if (current !== generation.current) return;
      const next = await api.get("components.workflows");
      if (current !== generation.current) return;
      setSnapshot(next);
      setNotice(success);
    } catch (reason) {
      if (current === generation.current)
        setError(
          `${applied ? "Change applied, but reloading failed." : operation ? "The change could not be confirmed." : "Unable to load workflow settings."} Refresh before retrying. ${message(reason)}`,
        );
    } finally {
      if (current === generation.current) setBusy(false);
    }
  }

  React.useEffect(() => {
    setSnapshot(undefined);
    setWorkflows([]);
    void load();
    void discover();
    return () => {
      generation.current++;
      discoveryGeneration.current++;
    };
  }, [api, runner]);

  function apply(name: string, label: string, value: Json, permanent: boolean) {
    return load(
      () => api.set(`components.workflows.${name}`, value, permanent),
      `${label} applied as a ${permanent ? "permanent" : "temporary"} override.`,
    );
  }

  function reset(name: string, label: string) {
    return load(
      () => api.reset(`components.workflows.${name}`),
      `${label} reset to startup TOML.`,
    );
  }

  const values = object(snapshot?.value);
  return (
    <>
      <div className="settings-toolbar">
        <p>Defaults for new workflow runs.</p>
        <button
          className="settings-icon-button"
          title="Refresh settings and workflows"
          aria-label="Refresh settings"
          disabled={busy || discovering}
          onClick={() => {
            void load();
            void discover();
          }}
        >
          <SettingsIcon name="refresh" />
        </button>
      </div>
      {error && (
        <div className="error" role="alert">
          {error}
        </div>
      )}
      {notice && <p role="status">{notice}</p>}
      {busy && <p role="status">Loading workflow settings…</p>}
      {snapshot && (
        <>
          {snapshot.exists &&
            (snapshot.value === null ||
              typeof snapshot.value !== "object" ||
              Array.isArray(snapshot.value)) && (
              <div className="error" role="alert">
                Invalid workflow settings object. Inspect the configuration JSON
                below.
              </div>
            )}
          {groups.map((group) => (
            <fieldset key={group.name} disabled={busy}>
              <legend>{group.label}</legend>
              {group.name === "agent" && (
                <>
                  {discovering && <p role="status">Discovering workflows…</p>}
                  {discoveryError && <p role="alert">{discoveryError}</p>}
                  <AgentSelection
                    value={object(values.agent).agents}
                    workflows={workflows}
                    apply={(value, permanent) =>
                      apply("agent.agents", "Allowed agents", value, permanent)
                    }
                    reset={() => reset("agent.agents", "Allowed agents")}
                  />
                </>
              )}
              {group.fields.map((field) => (
                <controls.SettingField
                  key={field.name}
                  field={field}
                  value={object(values[group.name])[field.name]}
                  suggest={(prefix) =>
                    api.suggest(
                      `components.workflows.${group.name}.${field.name}`,
                      prefix,
                    )
                  }
                  apply={(value, permanent) =>
                    apply(
                      `${group.name}.${field.name}`,
                      field.label,
                      value,
                      permanent,
                    )
                  }
                  reset={() =>
                    reset(`${group.name}.${field.name}`, field.label)
                  }
                />
              ))}
            </fieldset>
          ))}
          <DebugOnly>
            {() => (
              <details>
                <summary>Configuration JSON</summary>
                <pre>
                  {snapshot.exists
                    ? JSON.stringify(snapshot.value, null, 2)
                    : "Not configured"}
                </pre>
              </details>
            )}
          </DebugOnly>
        </>
      )}
    </>
  );
}

function AgentSelection({
  value,
  workflows,
  apply,
  reset,
}: {
  value: Json | undefined;
  workflows: Workflow[];
  apply: (value: string[], permanent: boolean) => Promise<void>;
  reset: () => Promise<void>;
}) {
  const serialized = JSON.stringify(value);
  const valid =
    value === undefined ||
    (Array.isArray(value) &&
      value.every((item) => typeof item === "string") &&
      new Set(value).size === value.length);
  const [draft, setDraft] = React.useState<string[]>([]);
  const [permanent, setPermanent] = React.useState(false);
  React.useEffect(() => {
    setDraft(
      value === undefined
        ? ["builtin:helper"]
        : Array.isArray(value)
          ? [
              ...new Set(
                value.filter(
                  (item): item is string => typeof item === "string",
                ),
              ),
            ]
          : [],
    );
  }, [serialized]);

  return (
    <form
      className="settings-field"
      aria-label="Edit Allowed agents"
      onSubmit={(event) => {
        event.preventDefault();
        if (valid) void apply(draft, permanent);
      }}
    >
      <div className="settings-field-value">
        <WorkflowSelect value={draft} change={setDraft} workflows={workflows} />
        {!valid && (
          <p role="alert">
            Configured agents must be a list of unique workflow references.
            Reset to restore startup TOML.
          </p>
        )}
        {value === undefined && (
          <small>Not configured. The built-in default is builtin:helper.</small>
        )}
      </div>
      <controls.SettingActions
        label="Allowed agents"
        valid={valid}
        reset={reset}
        permanent={permanent}
        changeLifetime={setPermanent}
      />
    </form>
  );
}
