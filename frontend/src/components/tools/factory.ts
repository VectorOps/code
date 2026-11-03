import type { ComponentType } from "react";
import type { toolDescription } from "../../models/steps";

export type ToolView = ReturnType<typeof toolDescription>;
export type ToolProps = { tool: ToolView; debug?: boolean };
export type ToolFormatter = {
  summary: ComponentType<ToolProps>;
  activity?: { completed: string; running: string };
  hideActivityLabel?: boolean;
  hideIndicators?: boolean;
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
    const label = name
      .replace(/^mcp_(.+_.+)_[a-f0-9]{20}$/, "mcp_$1")
      .replace(/^mcp_discover_[a-f0-9]{24}$/, "mcp_discover")
      .replace(/^mcp_(prompt|resource)_[a-f0-9]{32}$/, "mcp_$1");
    return (
      this.names[name] ??
      label
        .replace(/[_ .-]+/g, " ")
        .replace(/\b\w/g, (letter) => letter.toUpperCase())
        .replace(/\bMcp\b/g, "MCP")
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
