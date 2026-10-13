import { describe, expect, it } from "vitest";
import { workflowProjection } from "../src/models/workflow-tree";
import type { HistoryItem } from "../src/models/runner";
import type { Json } from "../src/models/json";

function drain(position: number, result: Json): HistoryItem {
  return {
    position,
    step: {
      id: `step-${position}`,
      workflow_id: "root",
      operation: { kind: "mailbox.drain", name: "drain", input: {} },
      outcome: { kind: "completed", result },
    },
  };
}

describe("steering history projection", () => {
  it("projects consumed messages at their drain position", () => {
    const projection = workflowProjection(
      [drain(1, [{ content: [{ type: "text", text: "Change direction" }] }])],
      [],
    );

    expect(projection.roots[0].entries).toMatchObject([
      {
        kind: "mailbox.input",
        message: "Change direction",
        input: { request_type: "chat.message.v1", steering: true },
      },
    ]);
  });

  it("does not create chat entries for an empty drain", () => {
    const projection = workflowProjection([drain(1, [])], []);

    expect(projection.roots[0].entries).toEqual([]);
  });
});
