import type { RunnerConnection } from "./websocket";
import { ConfigurationApi } from "./configuration-api";

export type { ConfigurationSnapshot } from "./configuration-api";

export type LlmField =
  "model" | "temperature" | "max_output_tokens" | "reasoning.effort";

function fieldName(profile: string | null, field: LlmField): string {
  const target = ["components", "llm"];
  if (profile !== null) target.push("profiles", profile);
  return [...target, ...field.split(".")]
    .map((segment) => segment.replaceAll("~", "~0").replaceAll(".", "~1"))
    .join(".");
}

export class LlmSettingsApi {
  private configuration: ConfigurationApi;

  constructor(connection: RunnerConnection) {
    this.configuration = new ConfigurationApi(connection);
  }

  get() {
    return this.configuration.get("components.llm");
  }

  set(
    profile: string | null,
    field: LlmField,
    value: string | number,
    permanent: boolean,
  ) {
    return this.configuration.set(fieldName(profile, field), value, permanent);
  }

  reset(profile: string | null, field: LlmField) {
    return this.configuration.reset(fieldName(profile, field));
  }

  async suggest(profile: string | null, field: LlmField, prefix: string) {
    return this.configuration.suggest(fieldName(profile, field), prefix);
  }
}
