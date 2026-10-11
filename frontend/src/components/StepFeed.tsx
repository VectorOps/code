import { useEffect, useMemo, useRef, useState } from "react";
import type { HistoryItem, InputRequest } from "../models/runner";
import {
  workflowTree,
  workflowRequests,
  workflowPath,
} from "../models/workflow-tree";
import type { WorkflowNode } from "../models/workflow-tree";
import { Decision } from "./Decision";
import type { StepActions, StepRegistry } from "./steps/registry";
import "./workflow.css";

function requestId(request: InputRequest) {
  return `decision-${request.execution_id}-${request.request_id}`;
}

function InlineDecision({
  request,
  actions,
}: {
  request: InputRequest;
  actions: StepActions;
}) {
  const [value, setValue] = useState("");
  return (
    <section
      id={requestId(request)}
      tabIndex={-1}
      aria-label={`Input required: ${request.channel}`}
    >
      <Decision
        request={request}
        disabled={!actions.enabled}
        value={value}
        change={setValue}
        reply={(response, key) => actions.reply(request, response, key)}
      />
    </section>
  );
}

function WorkflowBranch({
  node,
  registry,
  actions,
  active,
}: {
  node: WorkflowNode;
  registry: StepRegistry;
  actions: StepActions;
  active: boolean;
}) {
  const details = useRef<HTMLDetailsElement>(null);
  const requests = workflowRequests(node);
  const signature = requests.map((request) => request.request_id).join("/");
  useEffect(() => {
    if (signature && details.current) details.current.open = true;
  }, [signature]);
  const running = active && node.status === "Unfinished";
  return (
    <details className="workflow-branch" ref={details}>
      <summary>
        <span>{node.label}</span>
        <span className="workflow-status">
          {requests.length ? (
            `Input required (${requests.length})`
          ) : running ? (
            <>
              <span className="workflow-spinner" aria-hidden="true" />
              Running
            </>
          ) : (
            node.status
          )}
        </span>
      </summary>
      <WorkflowContents
        node={node}
        registry={registry}
        actions={actions}
        active={active}
      />
    </details>
  );
}

function WorkflowContents({
  node,
  registry,
  actions,
  active,
}: {
  node: WorkflowNode;
  registry: StepRegistry;
  actions: StepActions;
  active: boolean;
}) {
  const inline = new Set<string>();
  return (
    <>
      {node.entries.map((entry) => {
        if ("entries" in entry)
          return (
            <WorkflowBranch
              key={entry.id}
              node={entry}
              registry={registry}
              actions={actions}
              active={active}
            />
          );
        const request = node.requests.find(
          (value) => value.request_id === entry.id,
        );
        if (
          request &&
          !(
            entry.kind === "llm.tool" &&
            request.request_type === "tool.approval.v1"
          )
        ) {
          inline.add(request.request_id);
          return (
            <InlineDecision
              key={entry.id}
              request={request}
              actions={actions}
            />
          );
        }
        const Component = registry.resolve(entry);
        if (Component) {
          if (request) inline.add(request.request_id);
          return (
            <section
              key={entry.id}
              id={request ? requestId(request) : undefined}
              tabIndex={request ? -1 : undefined}
            >
              <Component step={entry} actions={actions} />
            </section>
          );
        }
        return null;
      })}
      {node.requests
        .filter((request) => !inline.has(request.request_id))
        .map((request) => (
          <InlineDecision
            key={request.request_id}
            request={request}
            actions={actions}
          />
        ))}
    </>
  );
}

export function StepFeed({
  history,
  registry,
  actions,
  active,
}: {
  history: HistoryItem[];
  registry: StepRegistry;
  actions: StepActions;
  active: boolean;
}) {
  const roots = useMemo(
    () => workflowTree(history, actions.pending),
    [history, actions.pending],
  );
  const requests: { request: InputRequest; path: string }[] = [];
  function collect(node: WorkflowNode) {
    requests.push(
      ...node.requests.map((request) => ({
        request,
        path: workflowPath(node),
      })),
    );
    node.entries.forEach((entry) => {
      if ("entries" in entry) collect(entry);
    });
  }
  roots.forEach(collect);
  function review(request: InputRequest) {
    const target = document.getElementById(requestId(request));
    if (!target) return;
    let parent = target.parentElement;
    while (parent) {
      if (parent instanceof HTMLDetailsElement) parent.open = true;
      parent = parent.parentElement;
    }
    target.focus({ preventScroll: true });
    target.scrollIntoView({ block: "center", behavior: "smooth" });
  }
  return (
    <>
      {requests.length > 0 && (
        <aside
          className="workflow-attention"
          aria-label="Pending workflow decisions"
        >
          <strong role="status">Needs your input ({requests.length})</strong>
          {requests.map(({ request, path }) => (
            <div key={request.request_id}>
              <span>
                {request.channel}
                <small>{path}</small>
              </span>
              <button onClick={() => review(request)}>Review</button>
            </div>
          ))}
        </aside>
      )}
      {roots.map((node) => (
        <WorkflowContents
          key={node.id}
          node={node}
          registry={registry}
          actions={actions}
          active={active}
        />
      ))}
      {!roots.length && (
        <p className="empty">
          No messages yet. Internal workflow activity is available in Debug
          steps.
        </p>
      )}
    </>
  );
}
