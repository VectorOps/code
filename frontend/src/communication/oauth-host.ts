import type { ConnectionSettings } from "../models/runner";

declare global {
  interface Window {
    pywebview?: {
      api: {
        authenticate(
          endpoint: string,
          token: string,
          id: string,
        ): Promise<boolean>;
      };
    };
  }
}

export async function authenticate(
  connection: ConnectionSettings,
  id: string,
): Promise<boolean> {
  const bridge = window.pywebview?.api;
  if (bridge)
    return bridge.authenticate(connection.endpoint, connection.token, id);
  const response = await fetch("/__runner/oauth", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...connection, id }),
    signal: AbortSignal.timeout(660000),
  });
  if (
    !response.ok ||
    !response.headers.get("content-type")?.includes("application/json")
  )
    throw new Error(
      "Local OAuth host unavailable. Start the frontend through the harness and retry.",
    );
  const result: { success: boolean } = await response.json();
  return result.success === true;
}
