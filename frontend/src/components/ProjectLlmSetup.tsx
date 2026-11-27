import * as React from "react";
import { createPortal } from "react-dom";
import { LlmSettingsApi } from "../communication/llm-settings-api";
import type { RunnerConnection } from "../communication/websocket";
import { useFocusBlocker } from "../focus/FocusProvider";
import { object, text } from "../models/json";
import "./project-llm-setup.css";

const fields = [
  { key: "default", profile: null, label: "Default model" },
  { key: "strong", profile: "strong", label: "Strong model" },
  { key: "medium", profile: "medium", label: "Medium model" },
  { key: "weak", profile: "weak", label: "Weak model" },
] as const;

type Models = Record<(typeof fields)[number]["key"], string>;

export function ProjectLlmSetup({
  connection,
  connected,
}: {
  connection?: RunnerConnection;
  connected: boolean;
}) {
  const api = React.useMemo(
    () => (connection ? new LlmSettingsApi(connection) : undefined),
    [connection],
  );
  const [prompt, setPrompt] = React.useState<{
    api: LlmSettingsApi;
    open: boolean;
    models: Models;
  }>();

  React.useEffect(() => {
    if (!api || !connected || prompt?.api === api) return;
    let current = true;
    void api.get().then(
      (snapshot) => {
        if (!current || typeof snapshot?.exists !== "boolean") return;
        const settings = object(snapshot.value);
        const profiles = object(settings.profiles);
        const models: Models = {
          default: text(settings.model),
          strong: text(object(profiles.strong).model),
          medium: text(object(profiles.medium).model),
          weak: text(object(profiles.weak).model),
        };
        const complete = fields.every(({ key }) => models[key].trim());
        setPrompt({ api, open: !complete, models });
      },
      () => {},
    );
    return () => {
      current = false;
    };
  }, [api, connected, prompt]);

  if (!api || prompt?.api !== api || !prompt.open) return null;
  return (
    <SetupDialog
      api={api}
      initialModels={prompt.models}
      connected={connected}
      dismiss={() => setPrompt({ ...prompt, open: false })}
    />
  );
}

function SetupDialog({
  api,
  initialModels,
  connected,
  dismiss,
}: {
  api: LlmSettingsApi;
  initialModels: Models;
  connected: boolean;
  dismiss: () => void;
}) {
  const dialog = React.useRef<HTMLDialogElement>(null);
  const active = React.useRef(false);
  const id = React.useId();
  const [models, setModels] = React.useState(initialModels);
  const [saving, setSaving] = React.useState(false);
  const [error, setError] = React.useState("");
  useFocusBlocker(true, true);
  React.useLayoutEffect(() => {
    const element = dialog.current;
    active.current = true;
    element?.showModal();
    return () => {
      active.current = false;
      element?.close();
    };
  }, []);

  async function save(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (saving || !connected || !models.default.trim()) return;
    setSaving(true);
    setError("");
    try {
      for (const { key, profile } of fields) {
        const model = models[key].trim() || models.default.trim();
        await api.set(profile, "model", model, true);
        if (!active.current) return;
      }
      dismiss();
    } catch {
      if (active.current) {
        setError("Could not save all the models. Please try again.");
        setSaving(false);
      }
    }
  }

  return createPortal(
    <dialog
      ref={dialog}
      className="project-llm-setup"
      aria-labelledby={`${id}-title`}
      aria-describedby={`${id}-description`}
      onCancel={(event) => {
        event.preventDefault();
        if (!saving) dismiss();
      }}
      onKeyDown={(event) => event.stopPropagation()}
    >
      <h2 id={`${id}-title`}>Set up LLM models</h2>
      <p id={`${id}-description`}>
        Enter a default model as <code>provider/model</code>. Leave any named
        profile blank to copy the default model into it. All four models are
        saved for this project only.
      </p>
      <form onSubmit={save}>
        <fieldset disabled={saving}>
          <legend className="sr-only">Project models</legend>
          {fields.map(({ key, label }) => (
            <label key={key}>
              {label}
              <input
                type="text"
                autoFocus={key === "default"}
                required={key === "default"}
                placeholder={
                  key === "default" ? "provider/model" : "Use default"
                }
                value={models[key]}
                onChange={(event) =>
                  setModels({ ...models, [key]: event.target.value })
                }
              />
            </label>
          ))}
        </fieldset>
        <p>
          Set up provider access separately in Settings → LLM authentication.
        </p>
        {!connected && (
          <p role="status">Waiting for the project to reconnect…</p>
        )}
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
        <div className="project-llm-setup-actions">
          <button
            className="quiet-button"
            type="button"
            disabled={saving}
            onClick={dismiss}
          >
            Not now
          </button>
          <button
            type="submit"
            disabled={saving || !connected || !models.default.trim()}
          >
            {saving ? "Saving…" : "Save models"}
          </button>
        </div>
      </form>
    </dialog>,
    document.body,
  );
}
