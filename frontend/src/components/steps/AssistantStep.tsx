import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { HighlightedCode } from "../HighlightedCode";
import { object, text } from "../../models/json";
import type { StepProps } from "./registry";

export function AssistantStep({ step, actions }: StepProps) {
  const streaming = step.state === "streaming" || step.state === "pending";
  return (
    <section className="feed-entry assistant-step" aria-busy={streaming}>
      <button
        className="activity-heading"
        onClick={() => actions.inspect(step.id)}
      >
        <span>Assistant</span>
        <span className="entry-state">
          {streaming ? "Responding..." : step.state} / Details
        </span>
      </button>
      <div className="markdown-content">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          skipHtml
          components={{
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
          }}
        >
          {step.message}
        </ReactMarkdown>
      </div>
      {step.state === "failed" && (
        <p className="failure-text">
          {text(object(step.outcome.error).message)}
        </p>
      )}
    </section>
  );
}
