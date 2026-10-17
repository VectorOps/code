import * as React from "react";
import type { Json } from "../models/json";
import { SettingsIcon } from "./SettingsIcon";

export interface SettingDefinition {
  name: string;
  label: string;
  description?: string;
  numeric?: boolean;
  multiline?: boolean;
  allowEmpty?: boolean;
  min?: number;
  max?: number;
  step?: number | "any";
  pattern?: string;
  placeholder?: string;
}

export function OverrideLifetime({
  permanent,
  change,
  disabled,
  label,
}: {
  permanent: boolean;
  change: (value: boolean) => void;
  disabled?: boolean;
  label: string;
}) {
  const name = React.useId();
  return (
    <div className="settings-scope">
      <div
        className="settings-segmented"
        role="group"
        aria-label={`${label} override lifetime`}
      >
        {[
          { value: false, label: "Temporary", icon: "temporary" as const },
          { value: true, label: "Workspace", icon: "workspace" as const },
        ].map((option) => (
          <label
            key={option.label}
            title={
              option.value
                ? "Permanent override saved in this workspace"
                : "Temporary override until runner restart"
            }
          >
            <input
              type="radio"
              aria-label={option.label}
              name={name}
              checked={permanent === option.value}
              disabled={disabled}
              onChange={() => change(option.value)}
            />
            <span>
              <SettingsIcon name={option.icon} />
            </span>
          </label>
        ))}
      </div>
    </div>
  );
}

export function SettingActions({
  label,
  valid = true,
  reset,
  permanent,
  changeLifetime,
}: {
  label: string;
  valid?: boolean;
  reset: () => Promise<void>;
  permanent: boolean;
  changeLifetime: (value: boolean) => void;
}) {
  return (
    <div className="settings-field-actions">
      <OverrideLifetime
        label={label}
        permanent={permanent}
        change={changeLifetime}
      />
      <div className="settings-actions">
        <button
          className="settings-icon-button settings-apply"
          title={`Apply ${label.toLowerCase()}`}
          aria-label={`Apply ${label.toLowerCase()}`}
          disabled={!valid}
        >
          <SettingsIcon name="apply" />
        </button>
        <button
          className="settings-icon-button"
          aria-label={`Reset ${label.toLowerCase()} to TOML`}
          title="Remove overrides and restore startup TOML"
          type="button"
          onClick={() => void reset()}
        >
          <SettingsIcon name="reset" />
        </button>
      </div>
    </div>
  );
}

export function SettingField({
  field,
  value,
  apply,
  reset,
  suggest,
}: {
  field: SettingDefinition;
  value: Json | undefined;
  apply: (value: string | number, permanent: boolean) => Promise<void>;
  reset: () => Promise<void>;
  suggest?: (prefix: string) => Promise<string[]>;
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
    if (!suggest || field.numeric || field.multiline) return;
    let active = true;
    setSuggestions([]);
    setSuggestionError(false);
    const timer = setTimeout(() => {
      void suggest(draft).then(
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
  }, [suggest, field.numeric, field.multiline, draft]);

  const valid =
    (field.allowEmpty || draft.trim() !== "") &&
    (!field.numeric || Number.isFinite(Number(draft)));

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
        {field.description && (
          <small id={`${id}-description`}>{field.description}</small>
        )}
        {field.multiline ? (
          <textarea
            id={id}
            rows={3}
            value={draft}
            placeholder={field.placeholder}
            aria-describedby={
              field.description ? `${id}-description` : undefined
            }
            onChange={(event) => setDraft(event.target.value)}
          />
        ) : (
          <input
            id={id}
            type={field.numeric ? "number" : "text"}
            min={field.min}
            max={field.max}
            step={field.step}
            pattern={field.pattern}
            required={!field.allowEmpty}
            placeholder={field.placeholder}
            value={draft}
            aria-describedby={
              field.description ? `${id}-description` : undefined
            }
            list={suggest && !field.numeric ? `${id}-suggestions` : undefined}
            onChange={(event) => setDraft(event.target.value)}
          />
        )}
        <div className="settings-current">
          Configured:{" "}
          <code>
            {value === undefined ? "Not configured" : JSON.stringify(value)}
          </code>
        </div>
        {suggest && !field.numeric && (
          <datalist id={`${id}-suggestions`}>
            {suggestions.map((value) => (
              <option key={value} value={value} />
            ))}
          </datalist>
        )}
        {suggestionError && <small>Suggestions unavailable.</small>}
      </div>
      <SettingActions
        label={field.label}
        valid={Boolean(valid)}
        reset={reset}
        permanent={permanent}
        changeLifetime={setPermanent}
      />
    </form>
  );
}
