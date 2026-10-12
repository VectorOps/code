import type { RunnerConnection } from "./websocket";

export interface Profiles {
  active_profile: string;
  profiles: string[];
}

export interface Credential {
  provider: string;
  profile: string;
  status:
    | "missing"
    | "unreadable"
    | "present"
    | "expired"
    | "expiring"
    | "unknown_expiry";
  kind: "api_key" | "oauth2" | null;
  expires_at: number | null;
  has_refresh_token: boolean;
}

export interface Login {
  id: string;
  provider: string;
  profile: string;
  status: "waiting" | "claimed";
}

export class AuthApi {
  constructor(private connection: RunnerConnection) {}

  profiles() {
    return this.connection.request<Profiles>("auth.profile.list");
  }

  profile(action: "add" | "switch" | "delete", name: string) {
    return this.connection.request<Profiles>(`auth.profile.${action}`, {
      name,
    });
  }

  async credentials(profile: string) {
    const credentials: Credential[] = [];
    let after: string | null = null;
    do {
      const page: { credentials: Credential[]; next_after: string | null } =
        await this.connection.request("auth.credentials.list", {
          profile,
          after,
          limit: 100,
        });
      credentials.push(...page.credentials);
      after = page.next_after;
    } while (after);
    return credentials;
  }

  check(provider: string, profile: string) {
    return this.connection.request<Credential>("auth.credentials.check", {
      provider,
      profile,
    });
  }

  remove(provider: string, profile: string) {
    return this.connection.request("auth.credentials.delete", {
      provider,
      profile,
    });
  }

  provision(provider: string, profile: string, secret: string) {
    return this.connection.request("auth.provision", {
      provider,
      profile,
      kind: "api_key",
      secret,
    });
  }

  logins() {
    return this.connection.request<{ requests: Login[] }>("auth.list");
  }

  login(provider: string, profile: string) {
    return this.connection.request<Login>("auth.login", { provider, profile });
  }

  cancel(id: string) {
    return this.connection.request("auth.cancel", { id });
  }
}
