import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { memo } from "react";
import { HighlightedCode } from "./HighlightedCode";
import { CodeBlock } from "./CodeBlock";

const plugins = [remarkGfm];
const components: Components = {
  pre: ({ children }) => <CodeBlock>{children}</CodeBlock>,
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

export const MarkdownBody = memo(function MarkdownBody({
  message,
}: {
  message: string;
}) {
  return (
    <div className="markdown-content">
      <ReactMarkdown remarkPlugins={plugins} skipHtml components={components}>
        {message}
      </ReactMarkdown>
    </div>
  );
});
