import type { RunnerConnection } from "./websocket";
import type { Json } from "../models/json";

export interface ConfigurationSnapshot {
  name: string;
  exists: boolean;
  value: Json;
}

export class ConfigurationApi {
  constructor(private connection: RunnerConnection) {}

  get(name: string) {
    return this.connection.request<ConfigurationSnapshot>("configuration.get", {
      name,
    });
  }

  set(name: string, value: Json, permanent: boolean) {
    return this.connection.request<null>("configuration.set", {
      name,
      value,
      permanent,
    });
  }

  reset(name: string) {
    return this.connection.request<null>("configuration.reset", { name });
  }

  async suggest(name: string, prefix: string) {
    const result = await this.connection.request<{ values: Json[] }>(
      "configuration.suggest",
      { name, prefix },
    );
    return result.values.filter(
      (value): value is string => typeof value === "string",
    );
  }
}
