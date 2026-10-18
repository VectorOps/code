import type { Json } from "../models/json";
import { object } from "../models/json";
import type { InputRequest } from "../models/runner";
import { approvalToolDescription } from "../models/steps";
import { defaultToolFormatters } from "./tools/defaults";
import type { ToolFormatterFactory, ToolView } from "./tools/factory";
import { useRetainedState, useRetainedValue } from "./FeedState";

export function ApprovalActions({
  request,
  enabled,
  reply,
  tool = approvalToolDescription(request),
  tools = defaultToolFormatters,
}: {
  request: InputRequest;
  enabled: boolean;
  reply: (
    request: InputRequest,
    response: Json,
    key: string,
  ) => Promise<boolean>;
  tool?: ToolView;
  tools?: ToolFormatterFactory;
}) {
  const [submitted, setSubmitted] = useRetainedState(
    request,
    "approval-submitted",
    () => false,
  );
  const [sending, setSending] = useRetainedState(
    request,
    "approval-sending",
    () => false,
  );
  const inFlight = useRetainedValue(request, "approval-flight", () => ({
    current: false,
  }));
  const keys = useRetainedValue(request, "approval-keys", () => ({
    current: new Map<string, string>(),
  }));
  const properties = object(object(request.payload).auth_context).properties;
  const metadata = object(properties);
  const propertyNames = Object.keys(metadata).sort();
  const [selected, setSelected] = useRetainedState<string[]>(
    request,
    "approval-properties",
    () => propertyNames,
  );
  const Summary = tools.resolve(tool).summary;
  async function decide(
    decision: "approve" | "deny",
    scope: "call" | "workflow" | "workspace" = "call",
  ) {
    if (!enabled || submitted || inFlight.current) return;
    inFlight.current = true;
    setSending(true);
    const mask = Object.fromEntries(
      selected.flatMap((name) =>
        Object.hasOwn(metadata, name) ? [[name, metadata[name]]] : [],
      ),
    );
    const signature = JSON.stringify([decision, scope, mask]);
    const key = keys.current.get(signature) || crypto.randomUUID();
    keys.current.set(signature, key);
    try {
      if (
        await reply(
          request,
          { decision, scope, properties: scope === "call" ? {} : mask },
          key,
        )
      )
        setSubmitted(true);
    } finally {
      inFlight.current = false;
      setSending(false);
    }
  }
  return (
    <div className="approval-actions">
      <div className="approval-summary">
        <span>{tools.name(tool.name)}</span>
        <Summary tool={tool} />
      </div>
      {propertyNames.length > 0 && (
        <div className="approval-metadata">
          {propertyNames.map((name) => (
            <label key={name}>
              <input
                type="checkbox"
                role="switch"
                aria-label={`Match ${name}`}
                checked={selected.includes(name)}
                disabled={!enabled || submitted || sending}
                onChange={(event) =>
                  setSelected(
                    event.target.checked
                      ? [...new Set([...selected, name])]
                      : selected.filter((value) => value !== name),
                  )
                }
              />
              <span className="approval-switch" aria-hidden="true" />
              <span className="approval-metadata-key">{name}</span>
              <code className="approval-metadata-value">
                {JSON.stringify(metadata[name])}
              </code>
            </label>
          ))}
        </div>
      )}
      <div
        className="approval-buttons"
        onClick={(event) => event.stopPropagation()}
      >
        <span>{submitted ? "Decision acknowledged" : "Allow"}</span>
        <button
          className="approval-icon"
          aria-label="Approve once"
          title="Approve this invocation only"
          disabled={!enabled || submitted || sending}
          onClick={() => void decide("approve", "call")}
        >
          <ApprovalIcon scope="call" />
        </button>
        <button
          className="approval-icon"
          aria-label="Approve for workflow"
          title="Approve matching calls in this workflow"
          disabled={!enabled || submitted || sending}
          onClick={() => void decide("approve", "workflow")}
        >
          <ApprovalIcon scope="workflow" />
        </button>
        <button
          className="approval-icon"
          aria-label="Approve for workspace"
          title="Approve matching calls in this workspace"
          disabled={!enabled || submitted || sending}
          onClick={() => void decide("approve", "workspace")}
        >
          <ApprovalIcon scope="workspace" />
        </button>
        <button
          className="approval-icon approval-deny"
          aria-label="Deny"
          title="Deny this invocation"
          disabled={!enabled || submitted || sending}
          onClick={() => void decide("deny", "call")}
        >
          <svg aria-hidden="true" viewBox="0 0 24 24">
            <path d="m7 7 10 10M17 7 7 17" />
          </svg>
        </button>
      </div>
    </div>
  );
}

function ApprovalIcon({ scope }: { scope: "call" | "workflow" | "workspace" }) {
  if (scope === "call") return <span aria-hidden="true">1</span>;
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24">
      {scope === "workflow" ? (
        <path d="m5 12 4 4L19 6" />
      ) : (
        <>
          <path d="M3 7a2 2 0 0 1 2-2h5l2 2h7a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z" />
          <path d="m9 13 2 2 4-5" />
        </>
      )}
    </svg>
  );
}
