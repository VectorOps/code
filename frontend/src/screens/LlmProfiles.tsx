import { DebugOnly } from "../components/Diagnostics";
import * as React from "react";
import { LlmSettingsApi } from "../communication/llm-settings-api";
import type {
  ConfigurationSnapshot,
  LlmField,
} from "../communication/llm-settings-api";
import type { RunnerApi } from "../communication/runner-api";
import { SettingField } from "../components/SettingsControls";
import { SettingsIcon } from "../components/SettingsIcon";
import { object } from "../models/json";

const fields: {
  name: LlmField;
  label: string;
  numeric?: boolean;
  min?: number;
  max?: number;
  step?: number | "any";
  placeholder?: string;
}[] = [
  { name: "model", label: "Model", placeholder: "provider/model" },
  {
    name: "temperature",
    label: "Temperature",
    numeric: true,
    min: 0,
    max: 2,
    step: "any",
  },
  {
    name: "max_output_tokens",
    label: "Maximum output tokens",
    numeric: true,
    min: 1,
    step: 1,
  },
  { name: "reasoning.effort", label: "Reasoning effort" },
];

function message(reason: unknown): string {
  return reason instanceof Error ? reason.message : "Request failed.";
}

export function LlmProfiles({ api: runner }: { api: RunnerApi }) {
  const api = React.useMemo(
    () => new LlmSettingsApi(runner.connection),
    [runner],
  );
  const [snapshot, setSnapshot] = React.useState<ConfigurationSnapshot>();
  const [profile, setProfile] = React.useState<string | null>(null);
  const [busy, setBusy] = React.useState(true);
  const [error, setError] = React.useState("");
  const [notice, setNotice] = React.useState("");
  const generation = React.useRef(0);

  function accept(next: ConfigurationSnapshot) {
    setSnapshot(next);
    const catalog = object(object(next.value).profiles);
    setProfile((current) =>
      current !== null && Object.hasOwn(catalog, current) ? current : null,
    );
  }

  async function refresh() {
    const current = ++generation.current;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const next = await api.get();
      if (current !== generation.current) return;
      accept(next);
    } catch (reason) {
      if (current === generation.current)
        setError(`Unable to load LLM settings. ${message(reason)}`);
    } finally {
      if (current === generation.current) setBusy(false);
    }
  }

  React.useEffect(() => {
    setSnapshot(undefined);
    setProfile(null);
    void refresh();
    return () => {
      generation.current++;
    };
  }, [api]);

  async function change(operation: () => Promise<unknown>, notice: string) {
    const current = ++generation.current;
    let applied = false;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await operation();
      applied = true;
      if (current !== generation.current) return;
      const next = await api.get();
      if (current !== generation.current) return;
      accept(next);
      setNotice(notice);
    } catch (reason) {
      if (current === generation.current)
        setError(
          `${applied ? "Change applied, but reloading failed." : "The change could not be confirmed."} Refresh before retrying. ${message(reason)}`,
        );
    } finally {
      if (current === generation.current) setBusy(false);
    }
  }

  const values = object(snapshot?.value);
  const catalog = object(values.profiles);
  const profiles = Object.keys(catalog).sort();
  const target =
    profile === null
      ? snapshot?.exists
        ? snapshot.value
        : undefined
      : catalog[profile];
  const settings = object(target);
  const invalidTarget =
    target !== undefined &&
    (target === null || typeof target !== "object" || Array.isArray(target));

  return (
    <>
      {error && (
        <div className="error" role="alert">
          {error}
        </div>
      )}
      {notice && <p role="status">{notice}</p>}
      {busy && <p role="status">Loading LLM settings…</p>}
      <div className="settings-toolbar">
        {snapshot && (
          <label>
            Edit settings for
            <select
              disabled={busy}
              value={profile === null ? "shared" : `profile:${profile}`}
              onChange={(event) => {
                setProfile(
                  event.target.value === "shared"
                    ? null
                    : event.target.value.slice(8),
                );
                setNotice("");
              }}
            >
              <option value="shared">Shared settings</option>
              {profiles.map((name) => (
                <option key={name} value={`profile:${name}`}>
                  {name}
                </option>
              ))}
            </select>
          </label>
        )}
        <button
          className="settings-icon-button"
          title="Refresh settings"
          aria-label="Refresh settings"
          disabled={busy}
          onClick={() => void refresh()}
        >
          <SettingsIcon name="refresh" />
        </button>
      </div>
      {snapshot && (
        <>
          {!snapshot.exists && <p>No configured LLM settings.</p>}
          {invalidTarget && (
            <div className="error" role="alert">
              Invalid settings object.
            </div>
          )}
          {values.profiles !== undefined &&
            (values.profiles === null ||
              typeof values.profiles !== "object" ||
              Array.isArray(values.profiles)) && (
              <div className="error" role="alert">
                Invalid profile catalog.
              </div>
            )}
          <fieldset disabled={busy}>
            <legend>Configured values</legend>
            {fields.map((field) => (
              <SettingField
                key={`${profile === null ? "shared" : `profile:${profile}`}/${field.name}`}
                field={field}
                suggest={(prefix) => api.suggest(profile, field.name, prefix)}
                value={
                  field.name === "reasoning.effort"
                    ? object(settings.reasoning).effort
                    : settings[field.name]
                }
                apply={(value, permanent) =>
                  change(
                    () => api.set(profile, field.name, value, permanent),
                    `${field.label} applied as a ${permanent ? "permanent" : "temporary"} override.`,
                  )
                }
                reset={() =>
                  change(
                    () => api.reset(profile, field.name),
                    `${field.label} reset to startup TOML.`,
                  )
                }
              />
            ))}
          </fieldset>
          <DebugOnly>
            {() => (
              <details>
                <summary>Configuration JSON</summary>
                <pre>
                  {target === undefined
                    ? "Not configured"
                    : JSON.stringify(target, null, 2)}
                </pre>
              </details>
            )}
          </DebugOnly>
        </>
      )}
    </>
  );
}
