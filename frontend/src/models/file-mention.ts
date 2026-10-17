export type FileMention = { start: number; end: number; query: string };

export function fileMention(
  value: string,
  start: number,
  end = start,
): FileMention | undefined {
  if (start !== end) return;
  const match = /(?:^|[\s(])@([^\s@"\n]*|"(?:[^"\\\n]|\\.)*)$/u.exec(
    value.slice(0, start),
  );
  if (!match) return;
  const raw = match[1];
  const query = raw.startsWith('"')
    ? raw.slice(1).replace(/\\(.)/gu, "$1")
    : raw;
  const tail = value.slice(start);
  const continuation = raw.startsWith('"')
    ? /^(?:[^"\\\n]|\\.)*"?/u.exec(tail)?.[0]
    : /^[^\s@"\n]*/u.exec(tail)?.[0];
  return {
    start: start - raw.length - 1,
    end: start + (continuation?.length ?? 0),
    query,
  };
}

export function insertFileMention(
  value: string,
  mention: FileMention,
  path: string,
) {
  const reference = /[\s"\\]/u.test(path) ? JSON.stringify(path) : path;
  const suffix = value.slice(mention.end);
  const inserted = `@${reference}${/^\s/u.test(suffix) ? "" : " "}`;
  return {
    text: value.slice(0, mention.start) + inserted + suffix,
    caret: mention.start + inserted.length,
  };
}
