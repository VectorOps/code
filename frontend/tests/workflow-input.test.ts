import * as vitest from "vitest";
import * as input from "../src/models/workflow-input";
import type { Json } from "../src/models/json";

vitest.test(
  "referenced, nested, and complex union fields use the JSON fallback",
  () => {
    const schemas: Json[] = [
      { anyOf: [{ type: "string" }, { type: "integer" }] },
      { anyOf: [{ type: "object" }, { type: "null" }] },
      { anyOf: [{ $ref: "#/$defs/Config" }, { type: "null" }] },
      { anyOf: [{ type: "string" }, { type: "integer" }, { type: "null" }] },
      { oneOf: [{ type: "string" }, { type: "null" }] },
      { $ref: "#/$defs/Config" },
      { type: "object" },
      { type: "array", items: { type: "string" } },
      { type: ["string", "integer", "null"] },
    ];
    for (const field of schemas) {
      vitest
        .expect(input.workflowFields({ type: "object", properties: { field } }))
        .toBeNull();
    }
  },
);

vitest.test.each([
  ["string", "hello", "hello"],
  ["integer", 0, "0"],
  ["number", 1.5, "1.5"],
  ["boolean", false, "false"],
] as const)(
  "nullable %s fields default to null and require values when enabled",
  (type, value, draft) => {
    const properties: Json[] = [
      { anyOf: [{ type }, { type: "null" }], default: null },
      { anyOf: [{ type: "null" }, { type }], default: null },
      { type: [type, "null"], default: null },
    ];
    for (const property of properties) {
      const schema = { type: "object", properties: { value: property } };
      const fields = input.workflowFields(schema)!;
      vitest.expect(fields).not.toBeNull();
      vitest.expect(fields[0]).toMatchObject({ type, nullable: true });
      const defaults = input.workflowDefaults(schema);
      vitest
        .expect(input.fieldValues(fields, defaults))
        .toEqual({ value: null });
      vitest
        .expect(input.inputFromFields(fields, { value: null }, {}))
        .toEqual({ value: null });
      vitest.expect(input.fieldValues(fields, {})).toEqual({ value: null });
      vitest
        .expect(() => input.inputFromFields(fields, { value: "" }, defaults))
        .toThrow("required when enabled");
      vitest
        .expect(() =>
          input.inputFromFields(fields, { value: "" }, { value: "" }),
        )
        .toThrow("required when enabled");
      vitest
        .expect(input.fieldValues(fields, { value }))
        .toEqual({ value: draft });
      vitest
        .expect(input.inputFromFields(fields, { value: draft }, defaults))
        .toEqual({ value });
    }
  },
);

vitest.test("required nullable fields accept null but not omission", () => {
  const fields = input.workflowFields({
    type: "object",
    required: ["value"],
    properties: { value: { anyOf: [{ type: "integer" }, { type: "null" }] } },
  })!;
  vitest
    .expect(input.inputFromFields(fields, { value: null }, {}))
    .toEqual({ value: null });
  vitest
    .expect(() => input.inputFromFields(fields, { value: "" }, {}))
    .toThrow("required");
  vitest
    .expect(() => input.inputFromFields(fields, { value: "1.5" }, {}))
    .toThrow("integer");
});

vitest.test("nullable enums preserve scalar choices and explicit null", () => {
  const properties: Json[] = [
    { anyOf: [{ type: "string", enum: ["fast", "slow"] }, { type: "null" }] },
    { type: ["string", "null"], enum: ["fast", "slow", null] },
  ];
  for (const property of properties) {
    const fields = input.workflowFields({
      type: "object",
      properties: { mode: property },
    })!;
    vitest
      .expect(fields[0])
      .toMatchObject({ nullable: true, choices: ["fast", "slow"] });
    for (const mode of [null, "fast", "slow"]) {
      vitest
        .expect(
          input.inputFromFields(
            fields,
            input.fieldValues(fields, { mode }),
            {},
          ),
        )
        .toEqual({ mode });
    }
  }
});

vitest.test(
  "field conversion keeps falsy defaults and validates required and integer inputs",
  () => {
    const schema = {
      type: "object",
      required: ["count"],
      properties: {
        count: { type: "integer", default: 0 },
        enabled: { type: "boolean", default: false },
      },
    };
    const fields = input.workflowFields(schema)!;
    const defaults = input.workflowDefaults(schema);
    vitest.expect(defaults).toEqual({ count: 0, enabled: false });
    vitest
      .expect(
        input.inputFromFields(
          fields,
          input.fieldValues(fields, defaults),
          defaults,
        ),
      )
      .toEqual(defaults);
    vitest
      .expect(() => input.inputFromFields(fields, { count: "" }, {}))
      .toThrow("required");
    vitest
      .expect(() => input.inputFromFields(fields, { count: "1.2" }, {}))
      .toThrow("integer");
    vitest
      .expect(() => input.inputFromFields(fields, { count: "Infinity" }, {}))
      .toThrow("integer");
    vitest
      .expect(
        input.inputFromFields(fields, { count: "4", enabled: "" }, defaults),
      )
      .toEqual({ count: 4 });
  },
);

vitest.test(
  "raw values that cannot round-trip through form controls stay in JSON mode",
  () => {
    const fields = input.workflowFields({
      type: "object",
      properties: { value: { type: "string" } },
    })!;
    vitest
      .expect(() => input.fieldValues(fields, { value: null }))
      .toThrow("Keep editing as JSON");
    vitest
      .expect(() => input.inputFromFields(fields, { value: null }, {}))
      .toThrow("cannot be null");
    vitest
      .expect(input.inputFromFields(fields, { value: "" }, { value: "" }))
      .toEqual({ value: "" });
    vitest
      .expect(
        input.workflowFields({
          type: "object",
          properties: { value: { type: "string", default: null } },
        }),
      )
      .toBeNull();
  },
);
