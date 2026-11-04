import { object, text } from "./json";
import type { Json, Params } from "./json";

export type WorkflowField = {
  name: string;
  label: string;
  description: string;
  type: "string" | "number" | "integer" | "boolean";
  required: boolean;
  nullable: boolean;
  choices?: Json[];
};

export type WorkflowFieldValues = Record<string, string | null>;

export function workflowDefaults(schema: Json): Params {
  const defaults: Params = {};
  for (const [name, value] of Object.entries(
    object(object(schema).properties),
  )) {
    const field = object(value);
    if (field.default !== undefined) defaults[name] = field.default;
  }
  return defaults;
}

export function workflowFields(schema: Json): WorkflowField[] | null {
  const root = object(schema);
  if (
    root.type !== "object" ||
    root.$ref ||
    root.anyOf ||
    root.oneOf ||
    root.allOf
  )
    return null;
  const fields: WorkflowField[] = [];
  for (const [name, value] of Object.entries(object(root.properties))) {
    const field = object(value);
    if (field.$ref || field.oneOf || field.allOf) return null;
    let scalar = field;
    let nullable = false;
    if (field.anyOf) {
      if (!Array.isArray(field.anyOf) || field.anyOf.length !== 2) return null;
      const branches = field.anyOf.map(object);
      const nullBranch = branches.find(
        (branch) => branch.type === "null" && Object.keys(branch).length === 1,
      );
      if (!nullBranch || field.type || field.enum) return null;
      scalar = branches.find((branch) => branch !== nullBranch)!;
      nullable = true;
    }
    let type = scalar.type;
    if (Array.isArray(type) && type.length === 2 && type.includes("null")) {
      type = type.find((item) => item !== "null")!;
      nullable = true;
    }
    if (
      scalar.$ref ||
      scalar.oneOf ||
      scalar.allOf ||
      scalar.anyOf ||
      !["string", "number", "integer", "boolean"].includes(text(type))
    )
      return null;
    const choices = Array.isArray(scalar.enum) ? scalar.enum : undefined;
    if (choices?.some((item) => item !== null && typeof item === "object"))
      return null;
    if (choices && !field.anyOf) nullable &&= choices.includes(null);
    fields.push({
      name,
      label: text(field.title) || name,
      description: text(field.description),
      type: type as WorkflowField["type"],
      required: Array.isArray(root.required) && root.required.includes(name),
      nullable,
      choices: nullable ? choices?.filter((item) => item !== null) : choices,
    });
  }
  try {
    fieldValues(fields, workflowDefaults(schema));
    return fields;
  } catch {
    return null;
  }
}

export function fieldValues(
  fields: WorkflowField[],
  input: Params,
): WorkflowFieldValues {
  return Object.fromEntries(
    fields.map((field) => {
      const value = input[field.name];
      if ((value === null || value === undefined) && field.nullable)
        return [field.name, null];
      if (
        value !== undefined &&
        (field.choices
          ? !field.choices.includes(value)
          : typeof value !== (field.type === "integer" ? "number" : field.type))
      )
        throw new Error(
          `${field.label} cannot be represented by the form. Keep editing as JSON.`,
        );
      return [
        field.name,
        value === undefined
          ? ""
          : field.choices || field.type === "boolean"
            ? JSON.stringify(value)
            : String(value),
      ];
    }),
  );
}

export function inputFromFields(
  fields: WorkflowField[],
  values: WorkflowFieldValues,
  base: Params,
  validate = true,
): Params {
  const input = { ...base };
  for (const field of fields) {
    const value = values[field.name];
    if (value === null) {
      if (!field.nullable) throw new Error(`${field.label} cannot be null.`);
      input[field.name] = null;
    } else if (value === undefined || value === "") {
      if (validate && field.nullable)
        throw new Error(`${field.label} is required when enabled.`);
      if (base[field.name] === "") continue;
      if (validate && field.required)
        throw new Error(`${field.label} is required.`);
      delete input[field.name];
    } else if (field.choices || field.type === "boolean")
      input[field.name] = JSON.parse(value) as Json;
    else if (field.type === "string") input[field.name] = value;
    else {
      const number = Number(value);
      if (
        !Number.isFinite(number) ||
        (field.type === "integer" && !Number.isInteger(number))
      )
        throw new Error(
          `${field.label} must be ${field.type === "integer" ? "an integer" : "a number"}.`,
        );
      input[field.name] = number;
    }
  }
  return input;
}

export function parseWorkflowInput(value: string): Params {
  const input: unknown = JSON.parse(value);
  if (!input || Array.isArray(input) || typeof input !== "object")
    throw new Error("Workflow input must be a JSON object.");
  return input as Params;
}
