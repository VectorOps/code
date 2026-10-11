import { isUserResponse } from "../../models/steps";
import { AssistantStep } from "./AssistantStep";
import { ToolStep } from "./ToolStep";
import { UserStep } from "./UserStep";
import { StepRegistry } from "./registry";

export const defaultStepRegistry = new StepRegistry([
  {
    id: "assistant",
    matches: (step) => step.kind === "llm.generate",
    visible: (step) =>
      step.state !== "completed" || step.message.trim().length > 0,
    component: AssistantStep,
  },
  {
    id: "tool",
    matches: (step) => step.kind === "llm.tool",
    component: ToolStep,
  },
  { id: "user", matches: isUserResponse, component: UserStep },
]);
