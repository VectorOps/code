import type { ComponentType } from "react";
import type { toolDescription } from "../../models/steps";

export type ToolView = ReturnType<typeof toolDescription>;
export type ToolProps = { tool: ToolView };
export type ToolFormatter = {
  summary: ComponentType<ToolProps>;
  details: ComponentType<ToolProps>;
  width: number;
  visible?: (tool: ToolView) => boolean;
};

export class ToolFormatterFactory {
  constructor(
    private readonly names: Readonly<Record<string, string>>,
    private readonly formatters: Readonly<Record<string, ToolFormatter>>,
    private readonly fallback: ToolFormatter,
  ) {}

  name(name: string): string {
    return (
      this.names[name] ??
      name
        .replace(/[_ .-]+/g, " ")
        .replace(/\b\w/g, (letter) => letter.toUpperCase())
    );
  }

  resolve(tool: ToolView): ToolFormatter {
    return this.formatters[tool.name] ?? this.fallback;
  }
}
