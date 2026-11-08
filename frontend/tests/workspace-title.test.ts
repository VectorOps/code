import * as testing from "@testing-library/react";
import * as vitest from "vitest";
import { RunnerApi } from "../src/communication/runner-api";
import { RunnerConnection } from "../src/communication/websocket";
import { useWorkspaceTitle } from "../src/hooks/useWorkspaceTitle";

vitest.afterEach(() => {
  testing.cleanup();
  vitest.vi.restoreAllMocks();
});

function client() {
  return new RunnerApi(new RunnerConnection("ws://localhost", "test"));
}

vitest.test(
  "sets the browser title from the connected runner workspace",
  async () => {
    const api = client();
    const workspace = "/projects/workspace with spaces/<project>";
    const request = vitest.vi
      .spyOn(api.connection, "request")
      .mockResolvedValue({ workspace });
    testing.renderHook(() => useWorkspaceTitle(api, true));
    await testing.waitFor(() =>
      vitest.expect(document.title).toBe(`VectorOps Code - ${workspace}`),
    );
    vitest.expect(request).toHaveBeenCalledExactlyOnceWith("runner.info");
  },
);

vitest.test(
  "loads the title on connection and refreshes it on reconnect",
  async () => {
    const api = client();
    const request = vitest.vi
      .spyOn(api.connection, "request")
      .mockResolvedValue({ workspace: "/projects/first" });
    const view = testing.renderHook(
      ({ connected }) => useWorkspaceTitle(api, connected),
      { initialProps: { connected: false } },
    );
    vitest.expect(document.title).toBe("VectorOps Code");
    vitest.expect(request).not.toHaveBeenCalled();
    view.rerender({ connected: true });
    await testing.waitFor(() =>
      vitest.expect(document.title).toBe("VectorOps Code - /projects/first"),
    );
    view.rerender({ connected: false });
    vitest.expect(document.title).toBe("VectorOps Code");
    request.mockResolvedValue({ workspace: "/projects/second" });
    view.rerender({ connected: true });
    await testing.waitFor(() =>
      vitest.expect(document.title).toBe("VectorOps Code - /projects/second"),
    );
    vitest.expect(request).toHaveBeenCalledTimes(2);
  },
);

vitest.test(
  "ignores stale workspace responses after switching runners",
  async () => {
    const first = client();
    const second = client();
    let finish!: (value: { workspace: string }) => void;
    vitest.vi.spyOn(first.connection, "request").mockReturnValue(
      new Promise((resolve) => {
        finish = resolve;
      }),
    );
    vitest.vi
      .spyOn(second.connection, "request")
      .mockResolvedValue({ workspace: "/projects/current" });
    const view = testing.renderHook(({ api }) => useWorkspaceTitle(api, true), {
      initialProps: { api: first },
    });
    view.rerender({ api: second });
    await testing.waitFor(() =>
      vitest.expect(document.title).toBe("VectorOps Code - /projects/current"),
    );
    await testing.act(async () => finish({ workspace: "/projects/stale" }));
    vitest.expect(document.title).toBe("VectorOps Code - /projects/current");
  },
);

vitest.test(
  "keeps the fallback title when workspace metadata is unavailable",
  async () => {
    const api = client();
    vitest.vi
      .spyOn(api.connection, "request")
      .mockRejectedValue(new Error("Connection lost"));
    const view = testing.renderHook(
      ({ api }: { api: RunnerApi | undefined }) => useWorkspaceTitle(api, true),
      { initialProps: { api: undefined as RunnerApi | undefined } },
    );
    vitest.expect(document.title).toBe("VectorOps Code");
    view.rerender({ api });
    await testing.act(async () => {});
    vitest.expect(document.title).toBe("VectorOps Code");
  },
);
