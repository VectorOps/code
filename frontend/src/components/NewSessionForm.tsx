import { useState } from "react";
import type { Params } from "../models/json";
import type { Workflow } from "../models/runner";

export type NewSessionDraft = { reference: string; input: string };
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
  start: (reference: string, input: Params) => Promise<boolean>;
}) {
  const [error, setError] = useState("");
  async function submit() {
    try {
      const input: unknown = JSON.parse(value.input);
      if (!input || Array.isArray(input) || typeof input !== "object")
        throw new Error("Workflow input must be a JSON object");
      setError("");
      await start(value.reference, input as Params);
    } catch (reason) {
      setError(String(reason));
    }
  }
  return (
    <div>
      <label>
        Workflow
        <select
          value={value.reference}
          onChange={(event) =>
            change({ ...value, reference: event.target.value })
          }
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
      <details>
        <summary>Input schema</summary>
        <pre>
          {JSON.stringify(
            workflows.find(
              (item) => item.descriptor.reference === value.reference,
            )?.descriptor.input_schema,
            null,
            2,
          )}
        </pre>
      </details>
      <label>
        Workflow input (JSON)
        <textarea
          rows={10}
          value={value.input}
          onChange={(event) => change({ ...value, input: event.target.value })}
        />
      </label>
      {error && <p role="alert">{error}</p>}
      <button
        disabled={!enabled || !value.reference}
        onClick={() => void submit()}
      >
        Start conversation
      </button>
      <p>
        Each conversation is one top-level workflow execution. Required
        responses appear in the composer or pending decisions panel.
      </p>
    </div>
  );
}
