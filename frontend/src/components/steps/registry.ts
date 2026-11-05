import type { ComponentType } from "react";
import type { Json } from "../../models/json";
import type { BranchAction, InputRequest } from "../../models/runner";
import type { StepView } from "../../models/steps";
import type { ToolFormatterFactory } from "../tools/factory";

export type StepActions = {
  tools?: ToolFormatterFactory;
  pending: InputRequest[];
  enabled: boolean;
  inspect: (id: string) => void;
  rootWorkflowId?: string;
  worktreeId?: string;
  branch?: (id: string, action: BranchAction) => Promise<boolean>;
  allowBranchAfter?: boolean;
  branchSuccessors?: ReadonlyMap<string, string>;
  reply: (
    request: InputRequest,
    response: Json,
    key: string,
  ) => Promise<boolean>;
};
export type StepProps = {
  debug?: boolean;
  step: StepView;
  actions: StepActions;
  failuresRecovered?: boolean;
};
export type StepRenderer = {
  id: string;
  matches: (step: StepView) => boolean;
  visible?: (step: StepView) => boolean;
  component: ComponentType<StepProps>;
};

export class StepRegistry {
  constructor(private readonly renderers: readonly StepRenderer[]) {
    if (
      new Set(renderers.map((renderer) => renderer.id)).size !==
      renderers.length
    )
      throw new Error("Step renderer IDs must be unique");
  }

  resolve(step: StepView): ComponentType<StepProps> | undefined {
    const renderer = this.renderers.find((renderer) => renderer.matches(step));
    if (!renderer || renderer.visible?.(step) === false) return undefined;
    return renderer.component;
  }

  extend(...renderers: StepRenderer[]): StepRegistry {
    const replaced = new Set(renderers.map((renderer) => renderer.id));
    return new StepRegistry([
      ...renderers,
      ...this.renderers.filter((renderer) => !replaced.has(renderer.id)),
    ]);
  }
}
