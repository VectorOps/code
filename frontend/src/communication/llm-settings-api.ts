import type { RunnerConnection } from "./websocket";
import type { Json } from "../models/json";
import { ConfigurationApi } from "./configuration-api";

export type { ConfigurationSnapshot } from "./configuration-api";

type GenerationField =
  "model" | "temperature" | "max_output_tokens" | "reasoning.effort";

export type LlmField =
  | GenerationField
  | "compaction"
  | `compaction.${
      | "context_window"
      | "trigger_threshold_ratio"
      | "keep_recent_tokens"
      | "keep_recent_ratio"
      | "keep_recent_messages"
      | "summary_target_ratio"
      | "summary_max_output_tokens"
      | "max_context_characters"
      | "additional_instructions"
      | "instruction"}`
  | `compaction.summary_settings.${GenerationField}`;

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
    value: Json,
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
