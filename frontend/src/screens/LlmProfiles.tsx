import { DebugOnly } from "../components/Diagnostics";
import * as React from "react";
import { LlmSettingsApi } from "../communication/llm-settings-api";
import type {
  ConfigurationSnapshot,
  LlmField,
} from "../communication/llm-settings-api";
import type { RunnerApi } from "../communication/runner-api";
import { SettingActions, SettingField } from "../components/SettingsControls";
import type { SettingDefinition } from "../components/SettingsControls";
import { SettingsIcon } from "../components/SettingsIcon";
import { object } from "../models/json";
import type { Json } from "../models/json";

type LlmSettingDefinition = SettingDefinition & {
  name: LlmField;
  section?: "summary" | "advanced";
};

const fields = [
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
] satisfies LlmSettingDefinition[];

const compactionFields: LlmSettingDefinition[] = [
  {
    name: "compaction.context_window",
    label: "Context window",
    description:
      "Token limit override; otherwise resolved by the runner from the model.",
    numeric: true,
    min: 1,
    step: 1,
  },
  {
    name: "compaction.trigger_threshold_ratio",
    label: "Compaction trigger ratio",
    description:
      "Fraction of the context window that triggers compaction (default: 0.8).",
    numeric: true,
    min: 0,
    exclusiveMin: 0,
    max: 1,
    step: "any",
  },
  {
    name: "compaction.keep_recent_tokens",
    label: "Keep recent tokens",
    description: "Explicit retention budget instead of the keep recent ratio.",
    numeric: true,
    min: 1,
    step: 1,
  },
  {
    name: "compaction.keep_recent_ratio",
    label: "Keep recent ratio",
    description:
      "Fraction retained (default: 0.25). Must be below the trigger ratio unless an explicit token budget or legacy character limit is configured.",
    numeric: true,
    min: 0,
    exclusiveMin: 0,
    max: 1,
    exclusiveMax: 1,
    step: "any",
  },
  {
    name: "compaction.keep_recent_messages",
    label: "Keep recent messages",
    description: "Minimum recent messages to preserve (default: 8).",
    numeric: true,
    min: 1,
    step: 1,
  },
  {
    name: "compaction.summary_target_ratio",
    section: "summary",
    label: "Summary target ratio",
    description:
      "Soft target as a fraction of the estimated tokens being summarized (default: 0.1).",
    numeric: true,
    min: 0,
    exclusiveMin: 0,
    max: 1,
    exclusiveMax: 1,
    step: "any",
  },
  {
    name: "compaction.summary_max_output_tokens",
    section: "summary",
    label: "Summary token limit",
    numeric: true,
    min: 1,
    step: 1,
  },
  {
    name: "compaction.additional_instructions",
    section: "summary",
    label: "Additional summary instructions",
    multiline: true,
  },
  {
    name: "compaction.instruction",
    section: "advanced",
    label: "Summary instruction",
    description: "Overrides the configurable summary instruction.",
    multiline: true,
  },
  {
    name: "compaction.max_context_characters",
    section: "advanced",
    label: "Legacy context character limit",
    description: "Legacy character-based compaction trigger.",
    numeric: true,
    min: 1,
    step: 1,
  },
];

const summaryFields: LlmSettingDefinition[] = fields
  .filter((field) => field.name !== "max_output_tokens")
  .map((field) => ({
    ...field,
    name: `compaction.summary_settings.${field.name}`,
    label: `Summary ${field.label.toLowerCase()}`,
  }));

function CompactionMode({
  value,
  apply,
  reset,
}: {
  value: Json | undefined;
  apply: (value: Json, permanent: boolean) => Promise<void>;
  reset: () => Promise<void>;
}) {
  const initial =
    value === undefined ? "unset" : value === null ? "disabled" : "enabled";
  const [mode, setMode] = React.useState(initial);
  const [permanent, setPermanent] = React.useState(false);
  const id = React.useId();
  React.useEffect(() => setMode(initial), [initial]);
  return (
    <form
      className="settings-field"
      data-edited={mode !== initial || permanent}
      aria-label="Edit Compaction"
      onSubmit={(event) => {
        event.preventDefault();
        if (mode !== "unset")
          void apply(mode === "disabled" ? null : object(value), permanent);
      }}
    >
      <div className="settings-field-description">
        <label htmlFor={id}>Compaction</label>
        <small>Summarize older messages to make room for new context.</small>
      </div>
      <div className="settings-field-value">
        <select
          id={id}
          value={mode}
          onChange={(event) => setMode(event.target.value)}
        >
          <option value="unset" disabled>
            Not configured
          </option>
          <option value="enabled">Enabled</option>
          <option value="disabled">Disabled</option>
        </select>
        {mode === "disabled" && value !== null && (
          <small>
            Applying Disabled replaces this target's compaction overrides.
          </small>
        )}
        <SettingActions
          label="Compaction"
          valid={mode !== "unset"}
          permanent={permanent}
          changeLifetime={setPermanent}
          reset={reset}
        />
      </div>
    </form>
  );
}

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

  function renderField(field: LlmSettingDefinition) {
    const value = field.name
      .split(".")
      .reduce<Json | undefined>(
        (current, segment) => object(current)[segment],
        settings,
      );
    return (
      <SettingField
        key={`${profile === null ? "shared" : `profile:${profile}`}/${field.name}`}
        field={field}
        suggest={
          field.numeric || field.multiline
            ? undefined
            : (prefix) => api.suggest(profile, field.name, prefix)
        }
        value={value}
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
    );
  }

  return (
    <>
      {error && (
        <div className="error" role="alert">
          {error}
        </div>
      )}
      {notice && (
        <p className="settings-notice" role="status">
          {notice}
        </p>
      )}
      {busy && <p role="status">Loading LLM settings…</p>}
      <div className="settings-toolbar llm-profile-toolbar">
        {snapshot && (
          <label>
            <span>Profile</span>
            <select
              aria-label="Edit settings for"
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
        <p className="settings-profile-hint">
          {profile === null
            ? "Shared settings are the baseline for every profile."
            : "Only this profile's overrides are shown. Unset fields inherit shared settings."}
        </p>
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
          <fieldset className="settings-card" disabled={busy}>
            <legend>Generation</legend>
            <p className="settings-section-description">
              Model and response preferences.
            </p>
            {fields.map(renderField)}
          </fieldset>
          <fieldset className="settings-card" disabled={busy}>
            <legend>Context compaction</legend>
            <p className="settings-section-description">
              Keep long conversations within the model's context window.
            </p>
            {settings.compaction !== undefined &&
              settings.compaction !== null &&
              (typeof settings.compaction !== "object" ||
                Array.isArray(settings.compaction)) && (
                <div className="error" role="alert">
                  Invalid compaction policy. Enable or reset compaction to
                  replace it.
                </div>
              )}
            <CompactionMode
              key={profile === null ? "shared" : `profile:${profile}`}
              value={settings.compaction}
              apply={(value, permanent) =>
                change(
                  () => api.set(profile, "compaction", value, permanent),
                  `Compaction applied as a ${permanent ? "permanent" : "temporary"} override.`,
                )
              }
              reset={() =>
                change(
                  () => api.reset(profile, "compaction"),
                  "Compaction reset to startup TOML.",
                )
              }
            />
            {settings.compaction === null && (
              <p className="settings-inline-note">
                Compaction is disabled here. Enable it to edit the policy.
              </p>
            )}
            <details className="settings-disclosure">
              <summary>
                <span>
                  Compaction policy
                  <small>When to summarize and how much context to keep</small>
                </span>
              </summary>
              <fieldset
                className="settings-disclosure-fields"
                aria-label="Compaction policy"
                disabled={settings.compaction === null}
              >
                {compactionFields
                  .filter((field) => !field.section)
                  .map(renderField)}
              </fieldset>
            </details>
            <details className="settings-disclosure">
              <summary>
                <span>
                  Summary generation settings
                  <small>
                    Summary size, model, and additional instructions
                  </small>
                </span>
              </summary>
              <fieldset
                className="settings-disclosure-fields"
                aria-label="Summary generation settings"
                disabled={settings.compaction === null}
              >
                {compactionFields
                  .filter((field) => field.section === "summary")
                  .map(renderField)}
                {summaryFields.map(renderField)}
              </fieldset>
            </details>
            <details className="settings-disclosure">
              <summary>
                <span>
                  Advanced compaction
                  <small>Base instruction and legacy compatibility</small>
                </span>
              </summary>
              <fieldset
                className="settings-disclosure-fields"
                aria-label="Advanced compaction"
                disabled={settings.compaction === null}
              >
                {compactionFields
                  .filter((field) => field.section === "advanced")
                  .map(renderField)}
              </fieldset>
            </details>
          </fieldset>
          <details className="settings-help">
            <summary>How saving and inheritance work</summary>
            <ul>
              <li>
                Apply saves one field. Temporary changes last until runner
                restart; Workspace changes are saved in this workspace.
              </li>
              <li>
                Reset removes overrides and restores startup TOML, not
                necessarily inheritance.
              </li>
              <li>
                Unset compaction uses shared or workflow defaults. Enabling uses
                the configured policy or an empty one. Editing a policy field
                configures compaction.
              </li>
              <li>
                Disabling writes null and replaces this target's compaction
                overrides.
              </li>
              <li>
                Workflows can override these settings. Defaults shown describe
                current runner behavior; older workflows may retain legacy
                defaults. The runner validates the combined policy when used.
              </li>
            </ul>
          </details>
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
