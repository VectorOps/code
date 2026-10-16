import * as React from "react";
import { LlmSettingsApi } from "../communication/llm-settings-api";
import type {
  ConfigurationSnapshot,
  LlmField,
} from "../communication/llm-settings-api";
import type { RunnerApi } from "../communication/runner-api";
import { object } from "../models/json";
import type { Json } from "../models/json";

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
        <button disabled={busy} onClick={() => void refresh()}>
          Refresh settings
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
                api={api}
                profile={profile}
                field={field}
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
          <details>
            <summary>Configuration JSON</summary>
            <pre>
              {target === undefined
                ? "Not configured"
                : JSON.stringify(target, null, 2)}
            </pre>
          </details>
        </>
      )}
    </>
  );
}

function SettingField({
  api,
  profile,
  field,
  value,
  apply,
  reset,
}: {
  api: LlmSettingsApi;
  profile: string | null;
  field: (typeof fields)[number];
  value: Json | undefined;
  apply: (value: string | number, permanent: boolean) => Promise<void>;
  reset: () => Promise<void>;
}) {
  const id = React.useId();
  const initial =
    typeof value === "string" || typeof value === "number" ? String(value) : "";
  const [draft, setDraft] = React.useState(initial);
  const [permanent, setPermanent] = React.useState(false);
  const [suggestions, setSuggestions] = React.useState<string[]>([]);
  const [suggestionError, setSuggestionError] = React.useState(false);

  React.useEffect(() => setDraft(initial), [initial]);

  React.useEffect(() => {
    if (field.numeric) return;
    let active = true;
    setSuggestions([]);
    setSuggestionError(false);
    const timer = setTimeout(() => {
      void api.suggest(profile, field.name, draft).then(
        (values) => {
          if (active) setSuggestions(values);
        },
        () => {
          if (active) setSuggestionError(true);
        },
      );
    }, 200);
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [api, profile, field.name, field.numeric, draft]);

  const valid =
    draft.trim() !== "" && (!field.numeric || Number.isFinite(Number(draft)));

  return (
    <form
      className="settings-field"
      aria-label={`Edit ${field.label}`}
      onSubmit={(event) => {
        event.preventDefault();
        if (valid) void apply(field.numeric ? Number(draft) : draft, permanent);
      }}
    >
      <div className="settings-field-value">
        <label htmlFor={id}>{field.label}</label>
        <input
          id={id}
          type={field.numeric ? "number" : "text"}
          min={field.min}
          max={field.max}
          step={field.step}
          required
          placeholder={field.placeholder}
          value={draft}
          list={field.numeric ? undefined : `${id}-suggestions`}
          onChange={(event) => setDraft(event.target.value)}
        />
        <div className="settings-current">
          Configured:{" "}
          <code>
            {value === undefined ? "Not configured" : JSON.stringify(value)}
          </code>
        </div>
        {!field.numeric && (
          <datalist id={`${id}-suggestions`}>
            {suggestions.map((value) => (
              <option key={value} value={value} />
            ))}
          </datalist>
        )}
        {suggestionError && <small>Suggestions unavailable.</small>}
      </div>
      <div>
        <label htmlFor={`${id}-lifetime`}>Apply as</label>
        <select
          id={`${id}-lifetime`}
          aria-label={`${field.label} override lifetime`}
          value={permanent ? "permanent" : "temporary"}
          onChange={(event) => setPermanent(event.target.value === "permanent")}
        >
          <option value="temporary">Temporary — until runner restart</option>
          <option value="permanent">Permanent — workspace override</option>
        </select>
      </div>
      <div className="settings-actions">
        <button
          aria-label={`Apply ${field.label.toLowerCase()}`}
          disabled={!valid}
        >
          Apply
        </button>
        <button
          aria-label={`Reset ${field.label.toLowerCase()} to TOML`}
          type="button"
          onClick={() => void reset()}
        >
          Reset to TOML
        </button>
      </div>
    </form>
  );
}
