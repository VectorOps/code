import * as React from "react";
import { AppearanceControl } from "../components/Appearance";
import type { RunnerApi } from "../communication/runner-api";
import type { ConnectionSettings } from "../models/runner";
import { Authentication } from "./Authentication";
import { LlmProfiles } from "./LlmProfiles";
import "./settings.css";

const categories = {
  appearance: "Visual layout",
  authentication: "LLM authentication",
  profiles: "LLM profiles",
};

type Category = keyof typeof categories;

export function Settings({
  api,
  connected,
  connection,
}: {
  api?: RunnerApi;
  connected: boolean;
  connection: ConnectionSettings;
}) {
  const [category, setCategory] = React.useState<Category>("appearance");
  const title = categories[category];
  return (
    <div className="settings-page">
      <nav aria-label="Settings categories" className="settings-nav">
        {(Object.keys(categories) as Category[]).map((value) => (
          <button
            key={value}
            aria-current={category === value ? "page" : undefined}
            onClick={() => setCategory(value)}
          >
            {categories[value]}
          </button>
        ))}
      </nav>
      <section className="settings-content" aria-label={title}>
        <h3>{title}</h3>
        {category === "appearance" ? (
          <AppearanceControl />
        ) : !connected || !api ? (
          <p role="status">Runner disconnected.</p>
        ) : category === "profiles" ? (
          <LlmProfiles
            key={`${connection.endpoint}/${connection.token}`}
            api={api}
          />
        ) : (
          <Authentication
            key={`${connection.endpoint}/${connection.token}`}
            api={api}
            connection={connection}
          />
        )}
      </section>
    </div>
  );
}
