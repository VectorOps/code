import type { ConnectionSettings } from "../models/runner";

export function readBootstrap(): ConnectionSettings {
  const fragment = new URLSearchParams(location.hash.slice(1));
  const element = document.getElementById("runner-bootstrap");
  const local: Partial<ConnectionSettings> = JSON.parse(
    element?.textContent || "{}",
  );
  element?.remove();
  history.replaceState(null, "", location.pathname + location.search);
  return {
    managed: fragment.get("managed") === "1",
    endpoint:
      fragment.get("endpoint") || local.endpoint || "ws://127.0.0.1:8765",
    token:
      fragment.get("token") ||
      (fragment.has("endpoint") ? "" : local.token) ||
      "",
  };
}
