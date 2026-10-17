import * as vitest from "vitest";
import * as input from "../src/models/workflow-input";
import type { Json } from "../src/models/json";

vitest.test(
  "nullable, referenced, nested, and union fields use the JSON fallback",
  () => {
    const schemas: Json[] = [
      { anyOf: [{ type: "string" }, { type: "null" }], default: null },
      { $ref: "#/$defs/Config" },
      { type: "object" },
      { type: "array", items: { type: "string" } },
      { type: ["string", "null"] },
    ];
    for (const field of schemas) {
      vitest
        .expect(input.workflowFields({ type: "object", properties: { field } }))
        .toBeNull();
    }
  },
);

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
