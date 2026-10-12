import "./style.css";
import "./workspace.css";
import "./steps.css";
import { createRoot } from "react-dom/client";
import { readBootstrap } from "./communication/bootstrap";
import { RunnerScreen } from "./screens/RunnerScreen";
import { AppearanceProvider } from "./components/Appearance";
import "./appearance.css";

createRoot(document.getElementById("root")!).render(
  <AppearanceProvider>
    <RunnerScreen initial={readBootstrap()} />
  </AppearanceProvider>,
);
