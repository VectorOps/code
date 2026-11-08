import { useEffect } from "react";
import type { RunnerApi } from "../communication/runner-api";

export function useWorkspaceTitle(
  api: RunnerApi | undefined,
  connected: boolean,
) {
  useEffect(() => {
    document.title = "VectorOps Code";
    if (!api || !connected) return;
    let active = true;
    void api.workspace().then(
      (workspace) => {
        if (active && workspace)
          document.title = `VectorOps Code - ${workspace}`;
      },
      () => {},
    );
    return () => {
      active = false;
      document.title = "VectorOps Code";
    };
  }, [api, connected]);
}
