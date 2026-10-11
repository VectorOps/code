import { object, text, type Json } from "../../models/json";
import type { ToolProps, ToolView } from "./factory";
import type { ReactNode } from "react";

function Summary({ items, fallback }: { items: string[]; fallback: string }) {
  const first = items[0] || fallback;
  const compact = first.replace(/\s+/g, " ");
  const shortened =
    compact.length > 120 ? `${compact.slice(0, 117)}...` : compact;
  return (
    <p className="tool-summary" title={items.join(" / ") || fallback}>
      {shortened}
      {items.length > 1 ? " ..." : ""}
    </p>
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
      <details>
        <summary>Result / output</summary>
        {children}
      </details>
    </>
  );
}

export function values(value: Json | undefined): Json[] {
  return Array.isArray(value) ? value : [];
}

export function payload(tool: ToolView): Json | undefined {
  const content = values(object(tool.output).content);
  const first = text(object(content[0]).text);
  if (!first) return tool.output;
  try {
    return JSON.parse(first) as Json;
  } catch {
    return first;
  }
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

function Code({ value }: { value: Json | undefined }) {
  return <pre className="tool-code">{display(value)}</pre>;
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

export function GenericSummary({ tool }: ToolProps) {
  const args = object(tool.arguments);
  const keys = Object.keys(args);
  return (
    <Summary
      items={keys.map((key) => `${key}: ${display(args[key])}`)}
      fallback="Open details to inspect this call"
    />
  );
}

export function GenericDetails({ tool }: ToolProps) {
  return (
    <RequestResult tool={tool}>
      <Result tool={tool} />
    </RequestResult>
  );
}

export function FileSummary({ tool }: ToolProps) {
  const args = object(tool.arguments),
    result = object(payload(tool));
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
      items={paths.length ? paths : [text(result.path)].filter(Boolean)}
      fallback="File details"
    />
  );
}

export function FileDetails({ tool }: ToolProps) {
  const result = object(payload(tool));
  if (!Object.keys(result).length) return <GenericDetails tool={tool} />;
  const { content, text: lines, ...metadata } = result;
  return (
    <RequestResult tool={tool}>
      <Fields data={metadata} />
      {result.truncated === true && (
        <p className="tool-notice">
          Partial content. More data is available beyond this range.
        </p>
      )}
      {result.encoding === "base64" ? (
        <details>
          <summary>Binary content (base64)</summary>
          <Code value={content} />
        </details>
      ) : content !== undefined || lines !== undefined ? (
        <Code value={content ?? lines} />
      ) : (
        <Result tool={tool} />
      )}
    </RequestResult>
  );
}

export function FilesDetails({ tool }: ToolProps) {
  const result = object(payload(tool));
  if (!Array.isArray(result.entries)) return <GenericDetails tool={tool} />;
  return (
    <RequestResult tool={tool}>
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
    </RequestResult>
  );
}

function command(tool: ToolView): string {
  const args = object(tool.arguments),
    cmd = object(args.command);
  return (
    text(args.command) ||
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

export function ProcessSummary({ tool }: ToolProps) {
  return (
    <Summary
      items={[command(tool)].filter(Boolean)}
      fallback="Process details"
    />
  );
}

export function ProcessDetails({ tool }: ToolProps) {
  const result = object(payload(tool)),
    args = object(tool.arguments);
  const { chunks, ...metadata } = result;
  const { command: requestedCommand, ...options } = args;
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
    <RequestResult
      tool={tool}
      request={
        <>
          {requestedCommand !== undefined && <Code value={command(tool)} />}
          {tool.arguments === undefined ? (
            <p>Arguments were not recorded in this history.</p>
          ) : (
            <Fields data={options} />
          )}
        </>
      }
    >
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
    </RequestResult>
  );
}

export function ProcessesDetails({ tool }: ToolProps) {
  const result = payload(tool);
  if (!Array.isArray(result)) return <GenericDetails tool={tool} />;
  return (
    <RequestResult tool={tool}>
      {!result.length && <p>No retained processes.</p>}
      {result.map((value, index) => (
        <section key={index}>
          <h3>{text(object(value).process_id)}</h3>
          <Fields data={object(value)} />
        </section>
      ))}
    </RequestResult>
  );
}

export function WebSummary({ tool }: ToolProps) {
  const args = object(tool.arguments),
    result = object(payload(tool));
  return (
    <Summary
      items={[
        `${text(args.method) || "GET"} ${text(args.url) || text(result.url) || "Web resource"}`,
      ]}
      fallback="Web resource"
    />
  );
}

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
  const items = todos(tool);
  if (!items.length) return <GenericDetails tool={tool} />;
  return (
    <RequestResult tool={tool}>
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
    </RequestResult>
  );
}

export function PatchSummary({ tool }: ToolProps) {
  const patch = text(object(tool.arguments).text);
  const files = patch
    .split("\n")
    .filter((line) =>
      /^(\*\*\* (Add|Update|Delete|Move to) File:|\*\*\* Move to:|\+\+\+ |--- )/.test(
        line,
      ),
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
}

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
            <pre className="tool-diff">
              {patch.split("\n").map((line, index) => (
                <span
                  key={index}
                  className={
                    line.startsWith("+")
                      ? "diff-add"
                      : line.startsWith("-")
                        ? "diff-remove"
                        : /^(\*\*\*|@@)/.test(line)
                          ? "diff-heading"
                          : ""
                  }
                >
                  {line || " "}
                </span>
              ))}
            </pre>
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
  const result = payload(tool),
    data = object(result);
  const message =
    tool.error ||
    text(data.error) ||
    (data.timed_out === true
      ? "The process exceeded its execution timeout."
      : typeof data.returncode === "number" && data.returncode !== 0
        ? `Process exited with code ${data.returncode}.`
        : typeof result === "string"
          ? result
          : "The tool reported an unsuccessful result. Review the details below.");
  return (
    <section className="tool-error" aria-label="Tool error">
      <h3>Tool could not complete successfully</h3>
      <p>{message.slice(0, 600)}</p>
      {message.length > 600 && (
        <details>
          <summary>Full error</summary>
          <Code value={message} />
        </details>
      )}
    </section>
  );
}
