import { useId, useState } from "react";
import type { Params } from "../models/json";
import type { Workflow } from "../models/runner";
import * as inputs from "../models/workflow-input";
import { WorktreeSelect } from "./WorktreeSelect";
import { useWorktrees } from "./Worktrees";

export type NewSessionDraft = {
  reference: string;
  input: string;
  fields?: inputs.WorkflowFieldValues;
  raw?: boolean;
  worktreeId?: string;
};
export function NewSessionForm({
  workflows,
  enabled,
  value,
  change,
  start,
}: {
  workflows: Workflow[];
  enabled: boolean;
  value: NewSessionDraft;
  change: (value: NewSessionDraft) => void;
  start: (
    reference: string,
    input: Params,
    worktreeId?: string,
  ) => Promise<boolean>;
}) {
  const formId = useId();
  const [error, setError] = useState("");
  const [creating, setCreating] = useState(false);
  const { records, loading, error: worktreeError, api } = useWorktrees();
  const invalidWorktree =
    Boolean(value.worktreeId) &&
    (!api ||
      loading ||
      Boolean(worktreeError) ||
      !records.some(
        (item) => item.id === value.worktreeId && item.status === "ready",
      ));
  const workflow = workflows.find(
    (item) => item.descriptor.reference === value.reference,
  );
  const fields = workflow
    ? inputs.workflowFields(workflow.descriptor.input_schema)
    : null;
  const raw = value.raw || fields === null;
  function values() {
    return (
      value.fields ??
      inputs.fieldValues(fields ?? [], inputs.parseWorkflowInput(value.input))
    );
  }
  function input(validate = true) {
    const base = inputs.parseWorkflowInput(value.input);
    return raw
      ? base
      : inputs.inputFromFields(fields ?? [], values(), base, validate);
  }
  function toggle() {
    try {
      const current = input(false);
      change({
        ...value,
        input: JSON.stringify(current, null, 2),
        fields: inputs.fieldValues(fields ?? [], current),
        raw: !raw,
      });
      setError("");
    } catch (reason) {
      setError(String(reason));
    }
  }
  async function submit() {
    if (!enabled || creating || invalidWorktree) return;
    try {
      setError("");
      if (value.worktreeId)
        await start(value.reference, input(), value.worktreeId);
      else await start(value.reference, input());
    } catch (reason) {
      setError(String(reason));
    }
  }
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        void submit();
      }}
    >
      <label>
        Workflow
        <select
          value={value.reference}
          onChange={(event) => {
            const workflow = workflows.find(
              (item) => item.descriptor.reference === event.target.value,
            );
            change({
              worktreeId: value.worktreeId,
              reference: event.target.value,
              input: JSON.stringify(
                inputs.workflowDefaults(
                  workflow?.descriptor.input_schema ?? {},
                ),
                null,
                2,
              ),
            });
            setError("");
          }}
        >
          <option value="">Choose a workflow</option>
          {workflows.map((item) => (
            <option
              key={item.descriptor.reference}
              value={item.descriptor.reference}
            >
              {item.descriptor.name}
            </option>
          ))}
        </select>
      </label>
      {workflow?.descriptor.description && (
        <p>{workflow.descriptor.description}</p>
      )}
      <WorktreeSelect
        value={value.worktreeId || ""}
        change={(worktreeId) => change({ ...value, worktreeId })}
        enabled={enabled}
        creating={creating}
        setCreating={setCreating}
      />
      {fields !== null && (
        <button type="button" className="quiet-button" onClick={toggle}>
          {raw ? "Use form fields" : "Edit as JSON"}
        </button>
      )}
      {!raw &&
        fields?.map((field) => {
          const fieldId = `${formId}-${field.name}`;
          const active = !field.nullable || values()[field.name] !== null;
          const required = field.required || field.nullable;
          return (
            <div key={field.name}>
              <label
                htmlFor={field.nullable ? `${fieldId}-enabled` : fieldId}
                className="flex items-center gap-2"
              >
                {field.nullable && (
                  <input
                    id={`${fieldId}-enabled`}
                    className="w-auto"
                    type="checkbox"
                    aria-label={`Set ${field.label}`}
                    checked={active}
                    onChange={(event) =>
                      change({
                        ...value,
                        fields: {
                          ...values(),
                          [field.name]: event.target.checked ? "" : null,
                        },
                      })
                    }
                  />
                )}
                <span id={`${fieldId}-label`}>
                  {field.label}
                  {field.required ? " (required)" : " (optional)"}
                </span>
              </label>
              {active &&
                (field.choices || field.type === "boolean" ? (
                  <select
                    id={fieldId}
                    aria-labelledby={`${fieldId}-label`}
                    required={required}
                    value={values()[field.name] ?? ""}
                    onChange={(event) =>
                      change({
                        ...value,
                        fields: {
                          ...values(),
                          [field.name]: event.target.value,
                        },
                      })
                    }
                  >
                    <option value="">
                      {required ? "Choose a value" : "Use workflow default"}
                    </option>
                    {(field.choices ?? [true, false]).map((choice) => (
                      <option
                        key={JSON.stringify(choice)}
                        value={JSON.stringify(choice)}
                      >
                        {typeof choice === "string"
                          ? choice
                          : JSON.stringify(choice)}
                      </option>
                    ))}
                  </select>
                ) : field.type === "string" ? (
                  <textarea
                    id={fieldId}
                    aria-labelledby={`${fieldId}-label`}
                    rows={2}
                    required={required}
                    value={values()[field.name] ?? ""}
                    onChange={(event) =>
                      change({
                        ...value,
                        fields: {
                          ...values(),
                          [field.name]: event.target.value,
                        },
                      })
                    }
                  />
                ) : (
                  <input
                    id={fieldId}
                    aria-labelledby={`${fieldId}-label`}
                    type="number"
                    step={field.type === "integer" ? 1 : "any"}
                    required={required}
                    value={values()[field.name] ?? ""}
                    onChange={(event) =>
                      change({
                        ...value,
                        fields: {
                          ...values(),
                          [field.name]: event.target.value,
                        },
                      })
                    }
                  />
                ))}
              {field.description && <small>{field.description}</small>}
            </div>
          );
        })}
      {!raw && fields?.length === 0 && (
        <p>No inputs are required. You can start this workflow as is.</p>
      )}
      {workflow && raw && (
        <>
          {fields === null && workflow && (
            <p>
              This workflow has complex inputs. Use the schema below as a guide,
              or edit its default values.
            </p>
          )}
          <label>
            Workflow input (JSON)
            <textarea
              rows={10}
              value={value.input}
              onChange={(event) =>
                change({
                  ...value,
                  input: event.target.value,
                  fields: undefined,
                })
              }
            />
          </label>
        </>
      )}
      <details>
        <summary>Input schema</summary>
        <pre>{JSON.stringify(workflow?.descriptor.input_schema, null, 2)}</pre>
      </details>
      {error && <p role="alert">{error}</p>}
      <button
        disabled={!enabled || !value.reference || creating || invalidWorktree}
      >
        Start conversation
      </button>
      <p>
        Each conversation is one top-level workflow execution. Required
        responses appear in the composer or pending decisions panel.
      </p>
    </form>
  );
}
