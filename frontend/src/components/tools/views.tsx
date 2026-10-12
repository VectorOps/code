import { object, text, type Json } from "../../models/json";
import { toolFailureMessage, toolPayload } from "../../models/steps";
import type { ToolProps, ToolView } from "./factory";
import { memo, type ReactNode } from "react";
import { HighlightedCode } from "../HighlightedCode";
import { LazyDetails } from "../LazyDetails";

function Summary({ items, fallback }: { items: string[]; fallback: string }) {
  const first = items[0] || fallback;
  const compact = first.replace(/\s+/g, " ");
  const shortened =
    compact.length > 120 ? `${compact.slice(0, 117)}...` : compact;
  return (
    <span
      className="tool-summary"
      title={
        items
          .map((item) => item.slice(0, 240))
          .join(" / ")
          .slice(0, 600) || fallback
      }
    >
      <code>
        {shortened}
        {items.length > 1 ? " ..." : ""}
      </code>
    </span>
  );
}

function RequestResult({
  tool,
  request,
  children,
}: ToolProps & { request?: ReactNode; children: ReactNode }) {
  return (
    <>
      <h3>Request</h3>
      {request ??
        (tool.arguments === undefined ? (
          <p>Arguments were not recorded in this history.</p>
        ) : (
          <Fields data={object(tool.arguments)} />
        ))}
      <LazyDetails summary="Result / output">{() => children}</LazyDetails>
    </>
  );
}

export function values(value: Json | undefined): Json[] {
  return Array.isArray(value) ? value : [];
}

export function payload(tool: ToolView): Json | undefined {
  return toolPayload(tool.output);
}

function display(value: Json | undefined): string {
  return typeof value === "string"
    ? value
    : (JSON.stringify(value, null, 2) ?? "");
}

export function Fields({ data }: { data: Record<string, Json> }) {
  return (
    <dl className="tool-fields">
      {Object.entries(data)
        .filter(([, value]) => value !== null)
        .map(([key, value]) => (
          <div key={key}>
            <dt>{key.replaceAll("_", " ")}</dt>
            <dd>{display(value)}</dd>
          </div>
        ))}
    </dl>
  );
}

function Code({
  value,
  language,
}: {
  value: Json | undefined;
  language?: string;
}) {
  return (
    <pre className="tool-code">
      <HighlightedCode
        language={language ?? (typeof value === "string" ? undefined : "json")}
      >
        {display(value)}
      </HighlightedCode>
    </pre>
  );
}

export function Result({ tool }: ToolProps) {
  const blocks = values(object(tool.output).content);
  if (!blocks.length)
    return (
      <p>
        {tool.state === "Authorized"
          ? "Approval gate completed. The final tool result is not available in this history record."
          : "No result recorded yet."}
      </p>
    );
  return (
    <>
      {blocks.map((value, index) => {
        const block = object(value);
        const mime = text(block.mime_type);
        if (
          block.type === "image" &&
          /^image\/(png|jpeg|gif|webp)$/.test(mime) &&
          text(block.data)
        )
          return (
            <img
              key={index}
              className="tool-image"
              src={`data:${mime};base64,${text(block.data)}`}
              alt="Tool result preview"
            />
          );
        let content: Json = block.text ?? value;
        if (typeof content === "string") {
          try {
            content = JSON.parse(content) as Json;
          } catch {
            content = text(block.text);
          }
        }
        return <Code key={index} value={content} />;
      })}
    </>
  );
}

export const GenericSummary = memo(function GenericSummary({
  tool,
}: ToolProps) {
  const args = object(tool.arguments);
  const keys = Object.keys(args);
  return (
    <Summary
      items={keys.slice(0, 3).map((key) => `${key}: ${preview(args[key])}`)}
      fallback="Open details to inspect this call"
    />
  );
});

function preview(value: Json | undefined): string {
  if (typeof value === "string") return value.slice(0, 240);
  if (Array.isArray(value)) return `[${value.length} items]`;
  if (value && typeof value === "object") return "{…}";
  return String(value ?? "");
}

export function GenericDetails({ tool }: ToolProps) {
  return (
    <RequestResult tool={tool}>
      <Result tool={tool} />
    </RequestResult>
  );
}

export const FileSummary = memo(function FileSummary({ tool }: ToolProps) {
  const args = object(tool.arguments);
  const paths = [
    args.pattern,
    args.path,
    ...values(args.patterns),
    ...values(args.paths),
  ]
    .map(text)
    .filter(Boolean);
  return (
    <Summary
      items={
        paths.length
          ? paths
          : [text(object(payload(tool)).path)].filter(Boolean)
      }
      fallback="File details"
    />
  );
});

export function FileDetails({ tool }: ToolProps) {
  return (
    <RequestResult tool={tool}>
      <FileOutput tool={tool} />
    </RequestResult>
  );
}

function FileOutput({ tool }: ToolProps) {
  const result = object(payload(tool));
  if (!Object.keys(result).length) return <Result tool={tool} />;
  const { content, text: lines, ...metadata } = result;
  return (
    <>
      <Fields data={metadata} />
      {result.truncated === true && (
        <p className="tool-notice">
          Partial content. More data is available beyond this range.
        </p>
      )}
      {result.encoding === "base64" ? (
        <LazyDetails summary="Binary content (base64)">
          {() => <Code value={content} />}
        </LazyDetails>
      ) : content !== undefined || lines !== undefined ? (
        <Code value={content ?? lines} />
      ) : (
        <Result tool={tool} />
      )}
    </>
  );
}

export function FilesDetails({ tool }: ToolProps) {
  return (
    <RequestResult tool={tool}>
      <FilesOutput tool={tool} />
    </RequestResult>
  );
}

function FilesOutput({ tool }: ToolProps) {
  const result = object(payload(tool));
  if (!Array.isArray(result.entries)) return <Result tool={tool} />;
  return (
    <>
      <div className="tool-table">
        <table>
          <thead>
            <tr>
              <th>Path</th>
              <th>Kind</th>
              <th>Bytes</th>
            </tr>
          </thead>
          <tbody>
            {result.entries.map((value, index) => {
              const entry = object(value);
              return (
                <tr key={index}>
                  <td>{text(entry.path)}</td>
                  <td>{text(entry.kind)}</td>
                  <td>{display(entry.size)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {!result.entries.length && <p>Directory is empty.</p>}
      {result.truncated === true && (
        <p className="tool-notice">
          Listing limit reached. Some entries are not shown.
        </p>
      )}
    </>
  );
}

function command(tool: ToolView): string {
  const args = object(tool.arguments),
    requested = args.command ?? tool.processCommand,
    cmd = object(requested);
  return (
    text(requested) ||
    text(cmd.script) ||
    values(cmd.argv)
      .map((value) => {
        const argument = text(value);
        return /^[a-zA-Z0-9_./:=@%-]+$/.test(argument)
          ? argument
          : `'${argument.replaceAll("'", "'\\''")}'`;
      })
      .join(" ") ||
    text(args.process_id) ||
    text(object(payload(tool)).process_id)
  );
}

export const ProcessSummary = memo(function ProcessSummary({
  tool,
}: ToolProps) {
  return (
    <Summary
      items={[command(tool)].filter(Boolean)}
      fallback="Process details"
    />
  );
});

export function ProcessDetails({ tool }: ToolProps) {
  const args = object(tool.arguments);
  const { command: requestedCommand, ...options } = args;
  return (
    <RequestResult
      tool={tool}
      request={
        <>
          {(requestedCommand !== undefined ||
            tool.processCommand !== undefined) && (
            <Code value={command(tool)} language="bash" />
          )}
          {tool.arguments === undefined ? (
            <p>Arguments were not recorded in this history.</p>
          ) : (
            <Fields data={options} />
          )}
        </>
      }
    >
      <ProcessOutput tool={tool} />
    </RequestResult>
  );
}

function ProcessOutput({ tool }: ToolProps) {
  const result = object(payload(tool));
  const { chunks, ...metadata } = result;
  const streams = new Map<
    string,
    { stream: string; encoding: string; parts: string[] }
  >();
  for (const value of values(chunks)) {
    const chunk = object(value);
    const stream = text(chunk.stream),
      encoding = text(chunk.encoding);
    const key = `${stream}/${encoding}`;
    const group = streams.get(key) ?? { stream, encoding, parts: [] };
    group.parts.push(text(chunk.data));
    streams.set(key, group);
  }
  return (
    <>
      <Fields data={metadata} />
      {result.timed_out === true && (
        <p className="tool-notice">Execution timed out.</p>
      )}
      {typeof result.dropped_bytes === "number" && result.dropped_bytes > 0 && (
        <p className="tool-notice">
          {result.dropped_bytes} bytes of output were dropped.
        </p>
      )}
      {[...streams.values()].map((chunk, index) => {
        return (
          <section key={index}>
            <h4>
              {chunk.stream}
              {chunk.encoding === "base64" ? " (base64)" : ""}
            </h4>
            <Code
              value={chunk.parts.join(chunk.encoding === "base64" ? "\n" : "")}
            />
          </section>
        );
      })}
      {chunks !== undefined && !values(chunks).length && <p>No new output.</p>}
      {chunks === undefined && <Result tool={tool} />}
    </>
  );
}

export function ProcessesDetails({ tool }: ToolProps) {
  return (
    <RequestResult tool={tool}>
      <ProcessesOutput tool={tool} />
    </RequestResult>
  );
}

function ProcessesOutput({ tool }: ToolProps) {
  const result = payload(tool);
  if (!Array.isArray(result)) return <Result tool={tool} />;
  return (
    <>
      {!result.length && <p>No retained processes.</p>}
      {result.map((value, index) => (
        <section key={index}>
          <h3>{text(object(value).process_id)}</h3>
          <Fields data={object(value)} />
        </section>
      ))}
    </>
  );
}

export const WebSummary = memo(function WebSummary({ tool }: ToolProps) {
  const args = object(tool.arguments);
  return (
    <Summary
      items={[
        `${text(args.method) || "GET"} ${text(args.url) || text(object(payload(tool)).url) || "Web resource"}`,
      ]}
      fallback="Web resource"
    />
  );
});

export function WebDetails({ tool }: ToolProps) {
  return <FileDetails tool={tool} />;
}

function todos(tool: ToolView): Json[] {
  return values(object(payload(tool)).todos ?? object(tool.arguments).todos);
}

export function PlanSummary({ tool }: ToolProps) {
  const items = values(object(tool.arguments).todos);
  return (
    <Summary
      items={items.map(
        (item) => text(object(item).title) || text(object(item).id),
      )}
      fallback="Plan update"
    />
  );
}

export function PlanDetails({ tool }: ToolProps) {
  return (
    <RequestResult tool={tool}>
      <PlanOutput tool={tool} />
    </RequestResult>
  );
}

function PlanOutput({ tool }: ToolProps) {
  const items = todos(tool);
  if (!items.length) return <Result tool={tool} />;
  return (
    <>
      <ol className="tool-plan">
        {items.map((value, index) => {
          const todo = object(value);
          return (
            <li key={index} data-state={text(todo.status)}>
              <span>{text(todo.title) || text(todo.id)}</span>
              <small>{text(todo.status).replaceAll("_", " ")}</small>
            </li>
          );
        })}
      </ol>
    </>
  );
}

function Diff({ lines }: { lines: string[] }) {
  return (
    <pre className="tool-diff">
      <HighlightedCode language="diff">{lines.join("\n")}</HighlightedCode>
    </pre>
  );
}

export const PatchSummary = memo(function PatchSummary({ tool }: ToolProps) {
  const patch = text(object(tool.arguments).text);
  const files = Array.from(
    patch.matchAll(
      /^(\*\*\* (Add|Update|Delete|Move to) File:|\*\*\* Move to:|\+\+\+ |--- )[^\n]*/gm,
    ),
    (match) => match[0],
  );
  return (
    <Summary
      items={files.map(
        (line) =>
          `${object(tool.arguments).reverse === true ? "Reverse patch / " : ""}${line.replace(/^(\*\*\* |\+\+\+ |--- )/, "")}`,
      )}
      fallback="Repository changes"
    />
  );
});

export function PatchDetails({ tool }: ToolProps) {
  const patch = text(object(tool.arguments).text);
  return (
    <RequestResult
      tool={tool}
      request={
        <>
          {object(tool.arguments).reverse === true && <p>Reverse patch</p>}
          <h3>Requested diff</h3>
          {patch ? (
            <Diff lines={patch.split("\n")} />
          ) : (
            <p>Patch text was not recorded in this history.</p>
          )}
        </>
      }
    >
      <Result tool={tool} />
      <p className="tool-notice">
        The diff shows requested changes, not proof they were applied. Check the
        result for successful files and partial failures before retrying.
      </p>
    </RequestResult>
  );
}

export function WorkflowSummary({ tool }: ToolProps) {
  return (
    <Summary
      items={[text(object(tool.arguments).reference)].filter(Boolean)}
      fallback="Child workflow"
    />
  );
}

export function ToolError({ tool }: ToolProps) {
  const message = toolFailureMessage(tool);
  return (
    <section className="tool-error" aria-label="Tool error">
      <h3>Tool could not complete successfully</h3>
      <p>{message.slice(0, 600)}</p>
      {message.length > 600 && (
        <LazyDetails summary="Full error">
          {() => <Code value={message} />}
        </LazyDetails>
      )}
    </section>
  );
}
