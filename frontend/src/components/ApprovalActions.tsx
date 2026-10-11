import { useRef, useState } from "react";
import type { Json } from "../models/json";
import type { InputRequest } from "../models/runner";

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
  const [submitted, setSubmitted] = useState(false);
  const [sending, setSending] = useState(false);
  const inFlight = useRef(false);
  const keys = useRef(new Map<string, string>());
  async function decide(
    approved: boolean,
    scope: "call" | "workflow" | "workspace",
  ) {
    if (!enabled || submitted || inFlight.current) return;
    inFlight.current = true;
    setSending(true);
    const signature = `${approved}/${scope}`;
    const key = keys.current.get(signature) || crypto.randomUUID();
    keys.current.set(signature, key);
    try {
      if (await reply(request, { approved, scope }, key)) setSubmitted(true);
    } finally {
      inFlight.current = false;
      setSending(false);
    }
  }
  return (
    <div className="approval-actions">
      <p>{submitted ? "Decision acknowledged" : "Approval required"}</p>
      <div onClick={(event) => event.stopPropagation()}>
        <button
          disabled={!enabled || submitted || sending}
          onClick={() => void decide(true, "call")}
        >
          Approve once
        </button>
        <button
          disabled={!enabled || submitted || sending}
          onClick={() => void decide(true, "workflow")}
        >
          Approve this tool for workflow
        </button>
        <button
          disabled={!enabled || submitted || sending}
          onClick={() => void decide(true, "workspace")}
        >
          Approve this tool for workspace
        </button>
        <button
          className="quiet-button"
          disabled={!enabled || submitted || sending}
          onClick={() => void decide(false, "call")}
        >
          Reject
        </button>
      </div>
      <small>
        Workspace approval saves permission for this tool across workflows and
        restarts. Workflow approval applies only within this workflow. Other
        pending calls still require a decision.
      </small>
    </div>
  );
}
