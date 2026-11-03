import type { ConnectionSettings } from "../models/runner";

export function ConnectionForm({
  value,
  change,
  connect,
}: {
  value: ConnectionSettings;
  change: (value: ConnectionSettings) => void;
  connect: () => void;
}) {
  return (
    <div className="connection">
      <label>
        WebSocket endpoint
        <input
          value={value.endpoint}
          onChange={(event) =>
            change({ ...value, endpoint: event.target.value, managed: false })
          }
        />
      </label>
      <label>
        Access token
        <input
          type="password"
          autoComplete="off"
          value={value.token}
          onChange={(event) => change({ ...value, token: event.target.value })}
        />
      </label>
      <button onClick={connect}>Connect</button>
    </div>
  );
}
