import { useEffect, useMemo, useRef, useState } from "react";
import { AppearanceControl } from "../components/Appearance";
import { AuthApi } from "../communication/auth-api";
import { authenticate } from "../communication/oauth-host";
import type * as auth from "../communication/auth-api";
import type { RunnerApi } from "../communication/runner-api";
import type { ConnectionSettings } from "../models/runner";
import "./settings.css";

export function Settings({
  api,
  connected,
  connection,
}: {
  api?: RunnerApi;
  connected: boolean;
  connection: ConnectionSettings;
}) {
  const [category, setCategory] = useState("appearance");
  return (
    <div className="settings-page">
      <nav
        aria-label="Settings categories"
        className="flex flex-col self-start gap-1 overflow-visible border-r border-neutral-200 pr-3 pb-0"
      >
        <button
          className="min-w-0 rounded-lg border-0 bg-transparent px-3 py-2.5 text-left text-sm font-medium text-neutral-600 hover:bg-neutral-100 hover:text-neutral-950 aria-[current=page]:bg-neutral-100 aria-[current=page]:text-neutral-950"
          aria-current={category === "appearance" ? "page" : undefined}
          onClick={() => setCategory("appearance")}
        >
          Visual layout
        </button>
        <button
          className="min-w-0 rounded-lg border-0 bg-transparent px-3 py-2.5 text-left text-sm font-medium text-neutral-600 hover:bg-neutral-100 hover:text-neutral-950 aria-[current=page]:bg-neutral-100 aria-[current=page]:text-neutral-950"
          aria-current={category === "authentication" ? "page" : undefined}
          onClick={() => setCategory("authentication")}
        >
          LLM authentication
        </button>
      </nav>
      <section className="settings-content">
        {category === "appearance" ? (
          <>
            <h3>Visual layout</h3>
            <AppearanceControl />
            <p>
              Choose a light or dark theme, or follow your system appearance.
              This preference is saved on this device.
            </p>
          </>
        ) : (
          <>
            <h3>LLM authentication</h3>
            {!connected || !api ? (
              <p role="status">Connect to a runner to manage authentication.</p>
            ) : (
              <Authentication
                key={`${connection.endpoint}/${connection.token}`}
                api={api}
                connection={connection}
              />
            )}
          </>
        )}
      </section>
    </div>
  );
}

function Authentication({
  api: runner,
  connection,
}: {
  api: RunnerApi;
  connection: ConnectionSettings;
}) {
  const api = useMemo(() => new AuthApi(runner.connection), [runner]);
  const [profiles, setProfiles] = useState<auth.Profiles>();
  const [profile, setProfile] = useState("");
  const [name, setName] = useState("");
  const [credentials, setCredentials] = useState<auth.Credential[]>([]);
  const [logins, setLogins] = useState<auth.Login[]>([]);
  const [provider, setProvider] = useState("openai");
  const [secret, setSecret] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [oauth, setOauth] = useState("");
  const [oauthNotice, setOauthNotice] = useState("");
  const alive = useRef(true);
  const generation = useRef(0);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      generation.current++;
    };
  }, []);

  async function refresh() {
    const current = ++generation.current;
    const listing = await api.profiles();
    const selected = listing.profiles.includes(profile)
      ? profile
      : listing.active_profile;
    const [entries, pending] = await Promise.all([
      api.credentials(selected),
      api.logins(),
    ]);
    if (!alive.current || current !== generation.current) return;
    setProfiles(listing);
    setProfile(selected);
    setCredentials(entries);
    setLogins(pending.requests);
  }

  useEffect(() => {
    let running = false;
    const poll = async () => {
      if (running) return;
      running = true;
      try {
        await refresh();
      } catch {
        if (alive.current)
          setError(
            "Unable to load authentication settings. Check the connection and runner protocol version.",
          );
      } finally {
        running = false;
      }
    };
    void poll();
    const timer = setInterval(() => void poll(), 5000);
    return () => {
      clearInterval(timer);
      generation.current++;
    };
  }, [api, profile]);

  async function action(operation: () => Promise<unknown>, message: string) {
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await operation();
      if (!alive.current) return;
      if (message) setNotice(message);
      await refresh();
    } catch {
      if (alive.current)
        setError(
          "The operation could not be confirmed. Refresh the status before retrying.",
        );
    } finally {
      if (alive.current) setBusy(false);
    }
  }

  async function authorize(login: auth.Login) {
    setOauth(login.id);
    setOauthNotice(
      "Opening authorization in your browser. Complete sign-in there, or cancel the pending login below.",
    );
    try {
      const success = await authenticate(connection, login.id);
      if (alive.current) {
        setOauthNotice(
          success
            ? "OAuth credentials saved."
            : "OAuth was cancelled or could not complete. Check status and retry the pending login.",
        );
        await refresh();
      }
    } catch (reason) {
      if (alive.current)
        setOauthNotice(
          reason instanceof Error
            ? reason.message
            : "The OAuth host could not complete login.",
        );
    } finally {
      if (alive.current) setOauth("");
    }
  }

  const pending = logins.filter((login) => login.profile === profile);
  return (
    <>
      <p>
        Credentials are stored by the runner, outside workspace history. Profile
        selection is shared by runners using the same credential store.
      </p>
      {error && (
        <div className="error" role="alert">
          {error}
        </div>
      )}
      {notice && <p role="status">{notice}</p>}
      <button
        disabled={busy}
        onClick={() => void action(refresh, "Status refreshed.")}
      >
        Refresh status
      </button>
      <fieldset disabled={busy || !profiles}>
        <legend>Profiles</legend>
        <label>
          Manage profile
          <select
            value={profile}
            onChange={(event) => {
              generation.current++;
              setProfile(event.target.value);
              setCredentials([]);
              setSecret("");
              setNotice("");
            }}
          >
            {profiles?.profiles.map((value) => (
              <option key={value} value={value}>
                {value}
                {value === profiles.active_profile ? " (active)" : ""}
              </option>
            ))}
          </select>
        </label>
        <div className="settings-actions">
          <button
            disabled={!profile || profile === profiles?.active_profile}
            onClick={() =>
              void action(
                () => api.profile("switch", profile),
                "Active profile changed.",
              )
            }
          >
            Make active
          </button>
          <button
            disabled={!profile || profile === "default"}
            onClick={() => {
              if (
                window.confirm(
                  `Delete profile ${profile} and all its local credentials? Remote tokens are not revoked. Cancel pending logins first.`,
                )
              )
                void action(
                  () => api.profile("delete", profile),
                  "Profile deleted.",
                );
            }}
          >
            Delete profile
          </button>
        </div>
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void action(async () => {
              await api.profile("add", name);
              setName("");
            }, "Profile added. Active selection is unchanged.");
          }}
        >
          <label>
            New profile
            <input
              required
              pattern="[A-Za-z0-9_-]{1,64}"
              maxLength={64}
              value={name}
              onChange={(event) => setName(event.target.value)}
            />
          </label>
          <button>Add profile</button>
        </form>
      </fieldset>
      <fieldset disabled={busy || !profile}>
        <legend>Stored credentials</legend>
        <p>
          Status reflects local storage and expiry, not provider acceptance.
          Deleting does not revoke remote tokens or clear credentials already in
          use.
        </p>
        {!credentials.length && <p>No stored credentials.</p>}
        {credentials.map((entry) => (
          <article key={entry.provider} className="credential-card">
            <h4>{entry.provider}</h4>
            <dl>
              <dt>Kind</dt>
              <dd>{entry.kind ?? "Unknown"}</dd>
              <dt>Local status</dt>
              <dd>{entry.status.replaceAll("_", " ")}</dd>
              <dt>Expires</dt>
              <dd>
                {entry.expires_at === null
                  ? "Not recorded"
                  : new Date(entry.expires_at * 1000).toLocaleString()}
              </dd>
              <dt>Refresh token</dt>
              <dd>{entry.has_refresh_token ? "Stored" : "Not stored"}</dd>
            </dl>
            <div className="settings-actions">
              <button
                onClick={() =>
                  void action(async () => {
                    const checked = await api.check(entry.provider, profile);
                    if (alive.current)
                      setNotice(
                        `${entry.provider}: ${checked.status.replaceAll("_", " ")}`,
                      );
                  }, "")
                }
              >
                Check status
              </button>
              <button
                onClick={() => {
                  setProvider(entry.provider);
                  setSecret("");
                }}
              >
                Replace key
              </button>
              <button
                onClick={() => {
                  if (
                    window.confirm(
                      `Delete ${entry.provider} from ${profile}? Cancel any pending login first to prevent it from saving credentials again. Remote tokens are not revoked.`,
                    )
                  )
                    void action(
                      () => api.remove(entry.provider, profile),
                      "Credential deleted.",
                    );
                }}
              >
                Delete
              </button>
            </div>
          </article>
        ))}
        <form
          onSubmit={(event) => {
            event.preventDefault();
            if (
              credentials.some((entry) => entry.provider === provider) &&
              !window.confirm(
                "Replace the entire stored credential with this API key?",
              )
            )
              return;
            const value = secret;
            setSecret("");
            void action(
              () => api.provision(provider, profile, value),
              "API key saved.",
            );
          }}
        >
          <h4>Import or replace API key</h4>
          <label>
            Provider
            <input
              required
              pattern="[a-z][a-z0-9_-]{0,63}"
              maxLength={64}
              value={provider}
              onChange={(event) => setProvider(event.target.value)}
            />
          </label>
          <label>
            API key
            <input
              type="password"
              required
              maxLength={65536}
              autoComplete="new-password"
              value={secret}
              onChange={(event) => setSecret(event.target.value)}
            />
          </label>
          <button>Save API key</button>
        </form>
      </fieldset>
      <fieldset disabled={busy || !profile}>
        <legend>OAuth2 login</legend>
        {oauthNotice && (
          <p role="status" aria-live="polite">
            {oauthNotice}
          </p>
        )}
        <p>
          ChatGPT browser authorization stores tokens in the runner. Existing
          OAuth credentials can refresh automatically when used.
        </p>
        <button
          disabled={Boolean(oauth)}
          onClick={() =>
            void action(async () => {
              const login = await api.login("chatgpt", profile);
              void authorize(login);
            }, "")
          }
        >
          Sign in with ChatGPT
        </button>
        {!pending.length && <p>No pending logins.</p>}
        {pending.map((login) => (
          <article key={login.id}>
            <h4>{login.provider}</h4>
            <p>
              {login.status} / {login.id}
            </p>
            <div className="settings-actions">
              <button
                disabled={Boolean(oauth) || login.status === "claimed"}
                onClick={() => void authorize(login)}
              >
                Continue in browser
              </button>
              <button
                onClick={() =>
                  void action(async () => {
                    await api.cancel(login.id);
                    setOauthNotice("Login cancelled.");
                  }, "Login cancelled.")
                }
              >
                Cancel login
              </button>
            </div>
          </article>
        ))}
      </fieldset>
    </>
  );
}
