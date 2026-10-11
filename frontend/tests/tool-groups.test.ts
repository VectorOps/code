import assert from "node:assert/strict";
import { test } from "node:test";
import { groupTools } from "../src/models/tool-groups";
import { stepView } from "../src/models/steps";
import type { InputRequest } from "../src/models/runner";
import type { WorkflowNode } from "../src/models/workflow-tree";

function tool(name: string, position: number, gate = false) {
  return stepView({
    position,
    step: {
      id: `tool-${position}`,
      workflow_id: "root",
      operation: {
        kind: "llm.tool",
        input: { call: { name, arguments: {} }, gate_only: gate },
      },
      outcome: { kind: "completed", result: { tool_name: name } },
    },
  });
}

test("only consecutive identical tools group; single calls stay standalone", () => {
  const alternating = ["read_file", "exec", "read_file", "exec"].map(
    (name, index) => tool(name, index),
  );
  assert.deepEqual(groupTools(alternating, []), alternating);
  const consecutive = ["read_file", "read_file", "exec", "exec"].map(
    (name, index) => tool(name, index),
  );
  const groups = groupTools(consecutive, []);
  assert.equal(groups.length, 2);
  assert.deepEqual(
    groups.map((group) => "steps" in group && group.steps),
    [consecutive.slice(0, 2), consecutive.slice(2)],
  );
});

test("configuration is opt-in and patch calls never group", () => {
  for (const name of ["list_files", "read_file", "read_files", "exec"]) {
    const entries = [tool(name, 1), tool(name, 2)];
    assert.equal(groupTools(entries, []).length, 1);
    assert.deepEqual(groupTools(entries, [], {}), entries);
    assert.deepEqual(groupTools(entries, [], { [name]: false }), entries);
  }
  const custom = [tool("custom", 1), tool("custom", 2)];
  assert.deepEqual(groupTools(custom, []), custom);
  assert.equal(groupTools(custom, [], { custom: true }).length, 1);
  const patches = [tool("apply_patch", 1), tool("apply_patch", 2)];
  assert.deepEqual(groupTools(patches, [], { apply_patch: true }), patches);
});

test("pending approval and historical approval gates break groups", () => {
  const entries = [tool("exec", 1), tool("exec", 2), tool("exec", 3)];
  const approval: InputRequest = {
    request_id: entries[1].id,
    execution_id: "execution",
    workflow_id: "root",
    request_type: "tool.approval.v1",
    channel: "approval",
    payload: {},
    response_type: "json",
  };
  assert.deepEqual(groupTools(entries, [approval]), entries);
  const gated = [entries[0], tool("exec", 2, true), entries[2]];
  assert.deepEqual(groupTools(gated, []), gated);
});

test("workflow branches and messages break groups", () => {
  const branch: WorkflowNode = {
    id: "child",
    label: "Child",
    entries: [],
    requests: [],
    status: "Completed",
  };
  const message = { ...tool("read_file", 2), kind: "llm.generate" };
  for (const boundary of [branch, message]) {
    const entries = [tool("read_file", 1), boundary, tool("read_file", 3)];
    assert.deepEqual(groupTools(entries, []), entries);
  }
});

test("growing groups retain identity and use current member state", () => {
  const entries = [tool("exec", 1), tool("exec", 2)];
  const initial = groupTools(entries, [])[0];
  const updated = { ...entries[1], state: "failed" };
  const next = groupTools([entries[0], updated, tool("exec", 3)], [])[0];
  assert.equal(next.id, initial.id);
  assert.ok("steps" in next);
  assert.equal(next.steps.length, 3);
  assert.equal(next.steps[1], updated);
});
