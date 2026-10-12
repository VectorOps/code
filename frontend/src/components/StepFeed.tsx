import * as React from "react";
import type { HistoryItem, InputRequest } from "../models/runner";
import * as workflows from "../models/workflow-tree";
import type { WorkflowNode } from "../models/workflow-tree";
import {
  groupTools,
  defaultToolGrouping,
  type ToolGrouping,
} from "../models/tool-groups";
import { ToolGroup } from "./ToolGroup";
import { Decision } from "./Decision";
import type { StepActions, StepRegistry } from "./steps/registry";
import "./workflow.css";
import * as feedState from "./FeedState";

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
  const [value, setValue] = feedState.useDecisionDraft(request);
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
  grouping,
}: {
  node: WorkflowNode;
  registry: StepRegistry;
  actions: StepActions;
  active: boolean;
  grouping: ToolGrouping;
}) {
  const requests = React.useMemo(
    () => workflows.workflowRequests(node),
    [node],
  );
  const [expanded, setExpanded] = feedState.useFeedExpansion(
    node.id,
    requests.length > 0,
  );
  const signature = requests.map((request) => request.request_id).join("/");
  React.useEffect(() => {
    if (signature) setExpanded(true);
  }, [signature, setExpanded]);
  const running = active && node.status === "Unfinished";
  return (
    <details
      className="workflow-branch feed-item"
      open={expanded}
      onToggle={(event) => {
        setExpanded(event.currentTarget.open);
      }}
    >
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
          ) : node.status === "Completed" ? null : (
            node.status
          )}
        </span>
      </summary>
      {expanded && (
        <WorkflowContents
          node={node}
          registry={registry}
          actions={actions}
          active={active}
          grouping={grouping}
        />
      )}
    </details>
  );
}

function WorkflowContents({
  node,
  registry,
  actions,
  active,
  grouping,
}: {
  node: WorkflowNode;
  registry: StepRegistry;
  actions: StepActions;
  active: boolean;
  grouping: ToolGrouping;
}) {
  const inline = new Set<string>();
  const entries = React.useMemo(
    () =>
      node.entries.filter(
        (entry) =>
          "entries" in entry ||
          entry.kind === "llm.tool" ||
          registry.resolve(entry) ||
          node.requests.some((request) => request.request_id === entry.id),
      ),
    [node, registry],
  );
  const grouped = React.useMemo(
    () => groupTools(entries, actions.pending, grouping),
    [entries, actions.pending, grouping],
  );
  const failuresRecovered = React.useMemo(
    () => workflows.workflowFailuresRecovered(node),
    [node],
  );
  return (
    <>
      {grouped.map((entry) => {
        if ("steps" in entry)
          return (
            <ToolGroup
              key={entry.id}
              group={entry}
              registry={registry}
              actions={actions}
              failuresRecovered={failuresRecovered}
            />
          );
        if ("entries" in entry)
          return (
            <WorkflowBranch
              key={entry.id}
              node={entry}
              registry={registry}
              actions={actions}
              active={active}
              grouping={grouping}
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
              className="feed-item"
              key={entry.id}
              id={request ? requestId(request) : undefined}
              tabIndex={request ? -1 : undefined}
            >
              <Component
                step={entry}
                actions={actions}
                failuresRecovered={failuresRecovered}
              />
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

type FeedProps = {
  history: HistoryItem[];
  registry: StepRegistry;
  actions: StepActions;
  active: boolean;
  grouping?: ToolGrouping;
  projection?: workflows.WorkflowProjection;
};

export const StepFeed = React.memo(function StepFeed(props: FeedProps) {
  return (
    <feedState.FeedStateProvider requests={props.actions.pending}>
      <FeedContents {...props} />
    </feedState.FeedStateProvider>
  );
});

function FeedContents({
  history,
  registry,
  actions,
  active,
  grouping = defaultToolGrouping,
  projection,
}: FeedProps) {
  const roots = React.useMemo(
    () => projection?.roots ?? workflows.workflowTree(history, actions.pending),
    [history, actions.pending, projection],
  );
  const requests = React.useMemo(() => {
    const requests: {
      request: InputRequest;
      path: string;
      node: WorkflowNode;
    }[] = [];
    function collect(node: WorkflowNode) {
      requests.push(
        ...node.requests.map((request) => ({
          request,
          path: workflows.workflowPath(node),
          node,
        })),
      );
      node.entries.forEach((entry) => {
        if ("entries" in entry) collect(entry);
      });
    }
    roots.forEach(collect);
    return requests;
  }, [roots]);
  const expand = feedState.useExpandWorkflows();
  const [reviewing, setReviewing] = React.useState<InputRequest | null>(null);
  React.useLayoutEffect(() => {
    if (!reviewing) return;
    const target = document.getElementById(requestId(reviewing));
    if (!target) return;
    target.focus({ preventScroll: true });
    target.scrollIntoView({ block: "center", behavior: "smooth" });
    setReviewing(null);
  }, [reviewing]);
  function review(request: InputRequest, node: WorkflowNode) {
    let ancestor: WorkflowNode | undefined = node;
    while (ancestor) {
      expand?.(ancestor.id, true);
      ancestor = ancestor.parent;
    }
    setReviewing(request);
  }
  return (
    <>
      {requests.length > 0 && (
        <aside
          className="workflow-attention"
          aria-label="Pending workflow decisions"
        >
          <strong role="status">Needs your input ({requests.length})</strong>
          {requests.map(({ request, path, node }) => (
            <div key={request.request_id}>
              <span>
                {request.channel}
                <small>{path}</small>
              </span>
              <button onClick={() => review(request, node)}>Review</button>
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
          grouping={grouping}
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
