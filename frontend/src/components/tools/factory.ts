import type { ComponentType } from "react";
import type { toolDescription } from "../../models/steps";

export type ToolView = ReturnType<typeof toolDescription>;
export type ToolProps = { tool: ToolView };
export type ToolFormatter = {
  summary: ComponentType<ToolProps>;
  activity?: { completed: string; running: string };
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

  activity(tool: ToolView): string {
    if (tool.state === "Authorized")
      return `Authorized ${this.name(tool.name)}`;
    if (tool.state === "cancelled") return `${this.name(tool.name)} cancelled`;
    const labels = this.resolve(tool).activity;
    return labels
      ? tool.state === "pending" || tool.state === "streaming"
        ? labels.running
        : labels.completed
      : this.name(tool.name);
  }
}
