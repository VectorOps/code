import { createRoot } from "react-dom/client";
import { readBootstrap } from "./communication/bootstrap";
import { RunnerScreen } from "./screens/RunnerScreen";
import "./style.css";
import "./workspace.css";
import "./steps.css";

createRoot(document.getElementById("root")!).render(
  <RunnerScreen initial={readBootstrap()} />,
);
