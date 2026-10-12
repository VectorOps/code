import hljs from "highlight.js/lib/common";
import { memo } from "react";
import "highlight.js/styles/github.css";

const maximumHighlightedLength = 20_000;

export const HighlightedCode = memo(function HighlightedCode({
  children,
  language,
  className = "",
}: {
  children: string;
  language?: string;
  className?: string;
}) {
  const name = language?.toLowerCase();
  const recognized = name && hljs.getLanguage(name);
  if (!recognized || children.length > maximumHighlightedLength)
    return <code className={className}>{children}</code>;
  return (
    <code
      className={`hljs ${className}`}
      dangerouslySetInnerHTML={{
        __html: hljs.highlight(children, {
          language: name,
          ignoreIllegals: true,
        }).value,
      }}
    />
  );
});
