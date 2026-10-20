import * as React from "react";
import { ConfigurationApi } from "../communication/configuration-api";
import type { RunnerApi } from "../communication/runner-api";
import { SettingsIcon } from "../components/SettingsIcon";
import type { Json } from "../models/json";
import type { ToolRule, ToolRuleInput, ToolRuleScope } from "../models/runner";

type PropertyDraft = {
  id: string;
  key: string;
  value: string;
  enabled: boolean;
};

function message(reason: unknown) {
  return reason instanceof Error ? reason.message : "Request failed.";
}

export function ToolPermissions({
  api,
  scope = { scope: "workspace" },
}: {
  api: RunnerApi;
  scope?: ToolRuleScope;
}) {
  const configuration = React.useMemo(
    () => new ConfigurationApi(api.connection),
    [api],
  );
  const [rules, setRules] = React.useState<ToolRule[]>([]);
  const [selected, setSelected] = React.useState("");
  const [toolName, setToolName] = React.useState("");
  const [decision, setDecision] = React.useState<"approve" | "deny">("approve");
  const [properties, setProperties] = React.useState<PropertyDraft[]>([]);
  const [editorOpen, setEditorOpen] = React.useState(false);
  const [approveAll, setApproveAll] = React.useState(false);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState("");
  const [notice, setNotice] = React.useState("");

  async function refresh() {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const [nextRules, setting] = await Promise.all([
        api.rules(scope),
        scope.scope === "workspace"
          ? configuration.get("tools.approval.approve_all")
          : Promise.resolve({ exists: false, value: false }),
      ]);
      setRules(nextRules);
      setApproveAll(setting.exists && setting.value === true);
    } catch (reason) {
      setError(`Unable to load tool permissions. ${message(reason)}`);
    } finally {
      setBusy(false);
    }
  }

  React.useEffect(() => {
    setRules([]);
    clear();
    void refresh();
  }, [api, configuration, JSON.stringify(scope)]);

  function input(): ToolRuleInput {
    const mask: Record<string, Json> = {};
    for (const property of properties) {
      if (!property.enabled) continue;
      const key = property.key.trim();
      if (!key) throw new Error("Property names cannot be empty.");
      if (Object.hasOwn(mask, key))
        throw new Error(`Property name '${key}' is duplicated.`);
      try {
        mask[key] = JSON.parse(property.value) as Json;
      } catch {
        throw new Error(`Property '${key}' must have a valid JSON value.`);
      }
    }
    return {
      tool_name: toolName.trim(),
      decision,
      properties: mask,
    };
  }

  async function change(operation: () => Promise<unknown>, success: string) {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await operation();
      setRules(await api.rules(scope));
      setNotice(success);
      return true;
    } catch (reason) {
      setError(message(reason));
      return false;
    } finally {
      setBusy(false);
    }
  }

  async function edit(ruleId: string) {
    setBusy(true);
    setError("");
    try {
      const rule = await api.rule(scope, ruleId);
      setSelected(rule.id);
      setToolName(rule.tool_name);
      setDecision(rule.decision);
      setProperties(
        Object.entries(rule.properties).map(([key, value]) => ({
          id: crypto.randomUUID(),
          key,
          value: JSON.stringify(value),
          enabled: true,
        })),
      );
      setEditorOpen(true);
    } catch (reason) {
      setError(message(reason));
    } finally {
      setBusy(false);
    }
  }

  function clear() {
    setSelected("");
    setToolName("");
    setDecision("approve");
    setProperties([]);
  }

  return (
    <div className="tool-permissions settings-content">
      <p>
        {scope.scope === "workspace"
          ? "Workspace rules apply across workflows and runner restarts."
          : "Workflow rules apply only to this workflow execution."}{" "}
        Rules with empty properties affect every call to that tool.
      </p>
      {error && (
        <div className="error" role="alert">
          {error}
        </div>
      )}
      {notice && <p role="status">{notice}</p>}
      <div className="settings-toolbar">
        <p>
          {scope.scope === "workspace"
            ? "Workspace permissions"
            : `Workflow ${scope.workflow_id}`}
        </p>
        <button
          className="settings-icon-button"
          title="Refresh permissions"
          aria-label="Refresh permissions"
          disabled={busy}
          onClick={() => void refresh()}
        >
          <SettingsIcon name="refresh" />
        </button>
      </div>
      {scope.scope === "workspace" && (
        <>
          <label className="permission-fallback">
            <input
              type="checkbox"
              checked={approveAll}
              disabled={busy}
              onChange={(event) => {
                const checked = event.target.checked;
                void change(
                  () =>
                    configuration.set(
                      "tools.approval.approve_all",
                      checked,
                      true,
                    ),
                  `Global approve all ${checked ? "enabled" : "disabled"}.`,
                ).then((changed) => {
                  if (changed) setApproveAll(checked);
                });
              }}
            />
            Approve all tool calls globally
          </label>
          <small>This fallback applies when no reusable rule matches.</small>
        </>
      )}
      <fieldset disabled={busy}>
        <legend>
          {scope.scope === "workspace" ? "Workspace" : "Workflow"} rules
        </legend>
        {!rules.length && <p>No authorization rules in this scope.</p>}
        {rules.map((rule) => (
          <article className="permission-rule" key={rule.id}>
            <div className="permission-rule-content">
              <div className="permission-rule-heading">
                <strong>{rule.tool_name}</strong>
                <span>{rule.decision}</span>
              </div>
              <PropertyMaskPreview value={rule.properties} />
            </div>
            <div className="settings-actions">
              <button onClick={() => void edit(rule.id)}>Edit</button>
              <button
                onClick={() =>
                  void change(
                    () => api.deleteRule(scope, rule.id),
                    `Rule for ${rule.tool_name} deleted.`,
                  )
                }
              >
                Delete
              </button>
            </div>
          </article>
        ))}
      </fieldset>
      <details
        open={editorOpen}
        onToggle={(event) => setEditorOpen(event.currentTarget.open)}
      >
        <summary>{selected ? "Edit rule" : "Create rule manually"}</summary>
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void change(
              () =>
                selected
                  ? api.updateRule(scope, selected, input())
                  : api.createRule(scope, input()),
              selected
                ? "Authorization rule updated."
                : "Authorization rule created.",
            ).then((changed) => {
              if (changed) {
                clear();
                setEditorOpen(false);
              }
            });
          }}
        >
          <p>
            <small>
              Use the exact tool name shown in an approval prompt. Approving a
              call for a workflow or workspace creates a rule automatically.
            </small>
          </p>
          <label>
            Tool name
            <input
              required
              value={toolName}
              onChange={(event) => setToolName(event.target.value)}
            />
          </label>
          <label>
            Decision
            <select
              value={decision}
              onChange={(event) =>
                setDecision(event.target.value as "approve" | "deny")
              }
            >
              <option value="approve">Approve</option>
              <option value="deny">Deny</option>
            </select>
          </label>
          <PropertyMaskEditor value={properties} change={setProperties} />
          <p>
            <small>
              Values use JSON syntax, such as "internal", true, or 3. Only
              enabled properties are matched. None enabled means every call to
              this tool.
            </small>
          </p>
          <div className="settings-actions">
            <button disabled={busy || !toolName.trim()}>Save rule</button>
            {selected && (
              <button
                type="button"
                onClick={() => {
                  clear();
                  setEditorOpen(false);
                }}
              >
                Cancel edit
              </button>
            )}
          </div>
        </form>
      </details>
    </div>
  );
}

function PropertyMaskPreview({ value }: { value: Record<string, Json> }) {
  const properties = Object.entries(value);
  if (!properties.length) return <small>All calls to this tool</small>;
  return (
    <dl className="permission-property-preview">
      {properties.map(([key, property]) => (
        <React.Fragment key={key}>
          <dt>
            <span className="permission-preview-match" aria-hidden="true">
              <span className="approval-switch" />
            </span>
            <span>{key}</span>
          </dt>
          <dd>
            <code>{JSON.stringify(property)}</code>
          </dd>
        </React.Fragment>
      ))}
    </dl>
  );
}

function PropertyMaskEditor({
  value,
  change,
}: {
  value: PropertyDraft[];
  change: React.Dispatch<React.SetStateAction<PropertyDraft[]>>;
}) {
  return (
    <fieldset className="permission-properties">
      <legend>Property mask</legend>
      <div className="permission-mask-grid">
        {value.map((property, index) => (
          <div className="permission-property" key={property.id}>
            <label className="permission-match">
              <input
                type="checkbox"
                role="switch"
                aria-label={`Match property ${index + 1}`}
                checked={property.enabled}
                onChange={(event) =>
                  change((current) =>
                    current.map((item) =>
                      item.id === property.id
                        ? { ...item, enabled: event.target.checked }
                        : item,
                    ),
                  )
                }
              />
              <span className="approval-switch" aria-hidden="true" />
            </label>
            <input
              aria-label={`Property ${index + 1} name`}
              value={property.key}
              onChange={(event) =>
                change((current) =>
                  current.map((item) =>
                    item.id === property.id
                      ? { ...item, key: event.target.value }
                      : item,
                  ),
                )
              }
            />
            <input
              aria-label={`Property ${index + 1} value`}
              value={property.value}
              onChange={(event) =>
                change((current) =>
                  current.map((item) =>
                    item.id === property.id
                      ? { ...item, value: event.target.value }
                      : item,
                  ),
                )
              }
            />
          </div>
        ))}
      </div>
      <button
        type="button"
        onClick={() =>
          change((current) => [
            ...current,
            { id: crypto.randomUUID(), key: "", value: '""', enabled: true },
          ])
        }
      >
        Add property
      </button>
    </fieldset>
  );
}
