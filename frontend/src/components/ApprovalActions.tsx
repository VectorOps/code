import type { Json } from "../models/json";
import { object, text } from "../models/json";
import type { InputRequest } from "../models/runner";
import { useRetainedState, useRetainedValue } from "./FeedState";

export function ApprovalActions({
  request,
  enabled,
  reply,
}: {
  request: InputRequest;
  enabled: boolean;
  reply: (
    request: InputRequest,
    response: Json,
    key: string,
  ) => Promise<boolean>;
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
  const [scope, setScope] = useRetainedState<"call" | "workflow" | "workspace">(
    request,
    "approval-scope",
    () => "call",
  );
  const [selected, setSelected] = useRetainedState<string[]>(
    request,
    "approval-properties",
    () => propertyNames,
  );
  async function decide(decision: "approve" | "deny") {
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
      <p>{submitted ? "Decision acknowledged" : "Approval required"}</p>
      <dl>
        <dt>Tool</dt>
        <dd>{text(object(object(request.payload).call).name) || "Unknown"}</dd>
        <dt>Arguments</dt>
        <dd>
          <pre>
            {JSON.stringify(
              object(object(request.payload).call).arguments,
              null,
              2,
            )}
          </pre>
        </dd>
        <dt>Authorization metadata</dt>
        <dd>
          <pre>{JSON.stringify(metadata, null, 2)}</pre>
        </dd>
      </dl>
      <fieldset disabled={!enabled || submitted || sending}>
        <legend>Permission scope</legend>
        {(["call", "workflow", "workspace"] as const).map((value) => (
          <label key={value}>
            <input
              type="radio"
              name={`approval-scope-${request.request_id}`}
              value={value}
              checked={scope === value}
              onChange={() => setScope(value)}
            />
            {value === "call"
              ? "This invocation only"
              : value === "workflow"
                ? "This workflow"
                : "This workspace"}
          </label>
        ))}
      </fieldset>
      {scope !== "call" && (
        <fieldset disabled={!enabled || submitted || sending}>
          <legend>Metadata remembered by this rule</legend>
          {propertyNames.length ? (
            propertyNames.map((name) => (
              <label key={name}>
                <input
                  type="checkbox"
                  checked={selected.includes(name)}
                  onChange={(event) =>
                    setSelected(
                      event.target.checked
                        ? [...new Set([...selected, name])]
                        : selected.filter((value) => value !== name),
                    )
                  }
                />
                {name}: {JSON.stringify(metadata[name])}
              </label>
            ))
          ) : (
            <p>No metadata is available for this call.</p>
          )}
          {!selected.length && (
            <p role="status">
              No metadata selected. This rule will affect every call to this
              tool in the selected scope.
            </p>
          )}
        </fieldset>
      )}
      <div onClick={(event) => event.stopPropagation()}>
        <button
          disabled={!enabled || submitted || sending}
          onClick={() => void decide("approve")}
        >
          Approve
        </button>
        <button
          className="quiet-button"
          disabled={!enabled || submitted || sending}
          onClick={() => void decide("deny")}
        >
          Deny
        </button>
      </div>
      <small>
        Workflow and workspace decisions create reusable rules. An empty
        metadata mask applies to every call to this tool.
      </small>
    </div>
  );
}
