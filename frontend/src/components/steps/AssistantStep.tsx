import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { memo } from "react";
import { HighlightedCode } from "../HighlightedCode";
import { object, text } from "../../models/json";
import type { StepProps } from "./registry";
import { StepIndicators } from "./StepIndicators";

const plugins = [remarkGfm];
const components: Components = {
  code: ({ children, className }) => (
    <HighlightedCode
      language={/\blanguage-([^\s]+)/.exec(className ?? "")?.[1]}
      className={className}
    >
      {String(children ?? "")}
    </HighlightedCode>
  ),
  img: ({ alt }) => <span>[Image: {alt || "attachment"}]</span>,
  a: ({ children, href }) => (
    <a href={href} target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  ),
};
const streamingComponents: Components = {
  ...components,
  code: ({ children, className }) => (
    <code className={className}>{children}</code>
  ),
};

const MarkdownBody = memo(function MarkdownBody({
  message,
  streaming,
}: {
  message: string;
  streaming: boolean;
}) {
  return (
    <div className="markdown-content">
      <ReactMarkdown
        remarkPlugins={plugins}
        skipHtml
        components={streaming ? streamingComponents : components}
      >
        {message}
      </ReactMarkdown>
    </div>
  );
});

export const AssistantStep = memo(function AssistantStep({
  step,
  actions,
}: StepProps) {
  const streaming = step.state === "streaming" || step.state === "pending";
  return (
    <section className="feed-entry assistant-step" aria-busy={streaming}>
      <div className="assistant-controls">
        {streaming && <span>{step.message ? "Writing" : "Thinking"}</span>}
        <StepIndicators
          state={step.state}
          inspect={() => actions.inspect(step.id)}
        />
      </div>
      <MarkdownBody message={step.message} streaming={streaming} />
      {step.state === "failed" && (
        <p className="failure-text">
          {text(object(step.outcome.error).message) ||
            "The agent could not finish its response."}
        </p>
      )}
    </section>
  );
});
