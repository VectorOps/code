import { DebugOnly } from "../components/Diagnostics";
import * as React from "react";
import { AuthApi } from "../communication/auth-api";
import { authenticate } from "../communication/oauth-host";
import type * as auth from "../communication/auth-api";
import type { RunnerApi } from "../communication/runner-api";
import type { ConnectionSettings } from "../models/runner";

export function Authentication({
  api: runner,
  connection,
}: {
  api: RunnerApi;
  connection: ConnectionSettings;
}) {
  const api = React.useMemo(() => new AuthApi(runner.connection), [runner]);
  const [profiles, setProfiles] = React.useState<auth.Profiles>();
  const [profile, setProfile] = React.useState("");
  const [name, setName] = React.useState("");
  const [credentials, setCredentials] = React.useState<auth.Credential[]>([]);
  const [logins, setLogins] = React.useState<auth.Login[]>([]);
  const [providers, setProviders] = React.useState<auth.Provider[]>([]);
  const [provider, setProvider] = React.useState("");
  const [secret, setSecret] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState("");
  const [notice, setNotice] = React.useState("");
  const [oauth, setOauth] = React.useState("");
  const [oauthNotice, setOauthNotice] = React.useState("");
  const alive = React.useRef(true);
  const generation = React.useRef(0);

  React.useEffect(() => {
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
    const [entries, pending, available] = await Promise.all([
      api.credentials(selected),
      api.logins(),
      api.providers(),
    ]);
    if (!alive.current || current !== generation.current) return;
    setProfiles(listing);
    setProfile(selected);
    setCredentials(entries);
    setLogins(pending.requests);
    setProviders(available.providers);
    setProvider((current) =>
      available.providers.some(
        (item) =>
          item.provider === current &&
          item.credential_kinds.includes("api_key"),
      )
        ? current
        : "",
    );
  }

  React.useEffect(() => {
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
    setOauthNotice("Browser sign-in in progress…");
    try {
      const success = await authenticate(connection, login.id);
      if (alive.current) {
        setOauthNotice(
          success
            ? "OAuth credentials saved."
            : "Browser sign-in did not complete.",
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
  const keyProviders = providers.filter((item) =>
    item.credential_kinds.includes("api_key"),
  );
  const oauthProviders = providers.filter((item) => item.oauth_login);
  return (
    <>
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
        <legend>Credential profiles</legend>
        <label>
          Credential profile
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
            Set active profile
          </button>
          <button
            disabled={!profile || profile === "default"}
            onClick={() => {
              if (
                window.confirm(
                  `Delete credential profile "${profile}" and its stored credentials?`,
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
            }, "Profile added.");
          }}
        >
          <label>
            New credential profile
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
                disabled={
                  !keyProviders.some((item) => item.provider === entry.provider)
                }
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
                      `Delete stored ${entry.provider} credentials from "${profile}"?`,
                    )
                  )
                    void action(
                      () => api.remove(entry.provider, profile),
                      "Credential deleted.",
                    );
                }}
              >
                Delete stored credential
              </button>
            </div>
          </article>
        ))}
        <form
          onSubmit={(event) => {
            event.preventDefault();
            if (!keyProviders.some((item) => item.provider === provider))
              return;
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
          <h4>API key</h4>
          <label>
            Provider
            <select
              required
              value={provider}
              disabled={!keyProviders.length}
              onChange={(event) => {
                setProvider(event.target.value);
                setSecret("");
              }}
            >
              <option value="" disabled>
                Select a provider
              </option>
              {keyProviders.map((item) => (
                <option key={item.provider} value={item.provider}>
                  {item.display_name}
                </option>
              ))}
            </select>
          </label>
          {!keyProviders.length && <p>No API-key providers available.</p>}
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
          <button disabled={!provider}>Save API key</button>
        </form>
      </fieldset>
      <fieldset disabled={busy || !profile}>
        <legend>Browser sign-in</legend>
        {oauthNotice && (
          <p role="status" aria-live="polite">
            {oauthNotice}
          </p>
        )}
        {oauthProviders.map((item) => (
          <button
            key={item.provider}
            disabled={Boolean(oauth)}
            onClick={() =>
              void action(async () => {
                const login = await api.login(item.provider, profile);
                void authorize(login);
              }, "")
            }
          >
            Sign in with {item.display_name}
          </button>
        ))}
        {!oauthProviders.length && <p>No browser-login providers available.</p>}
        {!pending.length && <p>No pending logins.</p>}
        {pending.map((login) => (
          <article key={login.id}>
            <h4>{login.provider}</h4>
            <p>
              {login.status}
              <DebugOnly>{() => <> / {login.id}</>}</DebugOnly>
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
