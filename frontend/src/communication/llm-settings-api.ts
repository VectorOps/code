import type { RunnerConnection } from "./websocket";
import type { Json } from "../models/json";

export type LlmField =
  "model" | "temperature" | "max_output_tokens" | "reasoning.effort";

export interface ConfigurationSnapshot {
  name: string;
  exists: boolean;
  value: Json;
}

function fieldName(profile: string | null, field: LlmField): string {
  const target = ["components", "llm"];
  if (profile !== null) target.push("profiles", profile);
  return [...target, ...field.split(".")]
    .map((segment) => segment.replaceAll("~", "~0").replaceAll(".", "~1"))
    .join(".");
}

export class LlmSettingsApi {
  constructor(private connection: RunnerConnection) {}

  get() {
    return this.connection.request<ConfigurationSnapshot>("configuration.get", {
      name: "components.llm",
    });
  }

  set(
    profile: string | null,
    field: LlmField,
    value: string | number,
    permanent: boolean,
  ) {
    return this.connection.request<null>("configuration.set", {
      name: fieldName(profile, field),
      value,
      permanent,
    });
  }

  reset(profile: string | null, field: LlmField) {
    return this.connection.request<null>("configuration.reset", {
      name: fieldName(profile, field),
    });
  }

  async suggest(profile: string | null, field: LlmField, prefix: string) {
    const result = await this.connection.request<{ values: Json[] }>(
      "configuration.suggest",
      { name: fieldName(profile, field), prefix },
    );
    return result.values.filter(
      (value): value is string => typeof value === "string",
    );
  }
}
