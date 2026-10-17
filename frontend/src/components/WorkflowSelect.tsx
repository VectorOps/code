import * as React from "react";
import type { Workflow } from "../models/runner";

export function WorkflowSelect({
  value,
  change,
  workflows,
}: {
  value: string[];
  change: (value: string[]) => void;
  workflows: Workflow[];
}) {
  const id = React.useId();
  const [query, setQuery] = React.useState("");
  const [open, setOpen] = React.useState(false);
  const choices = [
    ...new Map(
      workflows.map((item) => [item.descriptor.reference, item.descriptor]),
    ).values(),
  ]
    .sort((a, b) => a.reference.localeCompare(b.reference))
    .filter((item) =>
      `${item.reference} ${item.name}`
        .toLowerCase()
        .includes(query.toLowerCase()),
    );

  return (
    <div
      className="workflow-select"
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) setOpen(false);
      }}
      onKeyDown={(event) => {
        if (event.key === "Escape") {
          event.preventDefault();
          event.stopPropagation();
          document.getElementById(id)?.focus();
          setOpen(false);
        }
      }}
    >
      <label htmlFor={id}>Allowed agents</label>
      <small id={`${id}-help`}>
        Workflows the agent can delegate to. An empty selection disables
        delegation.
      </small>
      <div className="workflow-tags">
        {value.map((reference) => (
          <span className="workflow-tag" key={reference}>
            <span title={reference}>{reference}</span>
            <button
              type="button"
              aria-label={`Remove ${reference}`}
              onClick={() => change(value.filter((item) => item !== reference))}
            >
              ×
            </button>
          </span>
        ))}
        <input
          id={id}
          type="search"
          placeholder="Find workflows…"
          value={query}
          aria-describedby={`${id}-help`}
          aria-expanded={open}
          aria-controls={`${id}-choices`}
          autoComplete="off"
          onFocus={() => setOpen(true)}
          onChange={(event) => {
            setQuery(event.target.value);
            setOpen(true);
          }}
          onKeyDown={(event) => {
            if (event.key === "Enter") event.preventDefault();
            if (event.key === "ArrowDown") {
              event.preventDefault();
              setOpen(true);
              document
                .getElementById(`${id}-choices`)
                ?.querySelector("input")
                ?.focus();
            }
            if (event.key === "Backspace" && !query && value.length)
              change(value.slice(0, -1));
          }}
        />
      </div>
      {open && (
        <div
          id={`${id}-choices`}
          className="workflow-options"
          role="group"
          aria-label="Available workflows"
        >
          {choices.map((item) => (
            <label key={item.reference} title={item.description ?? undefined}>
              <input
                type="checkbox"
                checked={value.includes(item.reference)}
                onChange={(event) =>
                  change(
                    event.target.checked
                      ? [...value, item.reference]
                      : value.filter(
                          (reference) => reference !== item.reference,
                        ),
                  )
                }
              />
              <span>
                <span>{item.name}</span>
                <code>{item.reference}</code>
              </span>
            </label>
          ))}
          {!choices.length && <p>No matching workflows.</p>}
        </div>
      )}
    </div>
  );
}
