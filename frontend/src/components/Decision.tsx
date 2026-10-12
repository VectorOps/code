import type { Json } from "../models/json";
import { inputResponse, type InputRequest } from "../models/runner";
import { ApprovalActions } from "./ApprovalActions";
import { LazyDetails } from "./LazyDetails";
import { useRetainedState, useRetainedValue } from "./FeedState";
import { useFocusRegion } from "../focus/FocusProvider";

export function Decision({
  request,
  disabled,
  value,
  change,
  reply,
}: {
  request: InputRequest;
  disabled: boolean;
  value: string;
  change: (value: string) => void;
  reply: (response: Json, key: string) => Promise<boolean>;
}) {
  const focusRef = useFocusRegion<HTMLElement>("protected");
  const [error, setError] = useRetainedState(request, "error", () => "");
  const retry = useRetainedValue(request, "retry", () => ({
    current: { signature: "", key: "" },
  }));
  async function submit() {
    try {
      const response = inputResponse(request, value);
      const signature = JSON.stringify(response);
      if (retry.current.signature !== signature)
        retry.current = { signature, key: crypto.randomUUID() };
      setError("");
      if (await reply(response, retry.current.key)) change("");
    } catch (reason) {
      setError(String(reason));
    }
  }
  return (
    <section className="decision-card" ref={focusRef}>
      <h3>{request.channel}</h3>
      <small>Workflow {request.workflow_id?.slice(0, 8) || "unknown"}</small>
      <pre>{JSON.stringify(request.payload, null, 2)}</pre>
      {request.request_type === "tool.approval.v1" ? (
        <ApprovalActions
          request={request}
          enabled={!disabled}
          reply={(_, response, key) => reply(response, key)}
        />
      ) : (
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void submit();
          }}
        >
          <label>
            Response{" "}
            {request.request_type === "chat.message.v1"
              ? "(message)"
              : `(${request.response_type})`}
            <textarea
              rows={5}
              value={value}
              onChange={(event) => change(event.target.value)}
            />
          </label>
          {error && <p role="alert">{error}</p>}
          <button disabled={disabled}>Send response</button>
        </form>
      )}
      {request.response_schema && (
        <LazyDetails summary="Response schema">
          {() => <pre>{JSON.stringify(request.response_schema, null, 2)}</pre>}
        </LazyDetails>
      )}
    </section>
  );
}
