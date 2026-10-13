import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { test } from "vitest";

function lastRule(css: string, selector: string): string {
  const rules = Array.from(css.matchAll(/([^{}]+)\{([^{}]*)\}/g)).filter(
    (match) => match[1].split(",").some((name) => name.trim() === selector),
  );
  const rule = rules.at(-1);
  assert.ok(rule, `Missing CSS rule: ${selector}`);
  return rule[2];
}

test("frontend CSS preserves global layer order and appearance overrides", async () => {
  const { stdout } = await promisify(execFile)(
    process.execPath,
    [
      "--input-type=module",
      "-e",
      `import { build } from "vite";
      const result = await build({
        logLevel: "silent",
        build: { write: false, minify: false, cssMinify: false },
      });
      const styles = result.output
        .filter((entry) => entry.type === "asset" && entry.fileName.endsWith(".css"))
        .map((entry) => entry.source);
      process.stdout.write(JSON.stringify(styles));`,
    ],
    {
      maxBuffer: 1024 * 1024,
    },
  );
  const styles: unknown = JSON.parse(stdout);
  assert.ok(Array.isArray(styles) && styles.length > 0);
  const css = styles.join("\n");
  assert.match(css, /\.image-thumbnail/);
  assert.match(css, /\.tool-plan/);
  for (const selector of [
    ".user-step .message-text",
    ".composer-area textarea",
    ".assistant-step",
    ".markdown-content",
    ".markdown-content p",
  ]) {
    assert.match(lastRule(css, selector), /font-size:\s*0?\.9375rem/);
  }
  const layers = Array.from(css.matchAll(/@layer\s+([^;{]+)[;{]/g)).flatMap(
    (match) => match[1].split(",").map((name) => name.trim()),
  );
  const order = Array.from(new Set(layers)).filter((name) =>
    ["theme", "base", "components", "utilities"].includes(name),
  );
  assert.deepEqual(order, ["theme", "base", "components", "utilities"]);
  assert.match(
    lastRule(css, ".tool-summary"),
    /color:\s*var\(--color-neutral-600\)/,
  );
  for (const selector of [".tool-error", ".tool-notice"]) {
    const rule = lastRule(css, selector);
    assert.match(rule, /border-left-color:\s*light-dark\(#a16207,\s*#fbbf24\)/);
    assert.match(rule, /background:\s*light-dark\(#fffbeb,\s*#422006\)/);
  }
  for (const selector of [
    ".tool-fields > div",
    ".tool-table th",
    ".tool-table td",
  ]) {
    assert.match(
      lastRule(css, selector),
      /border-bottom-color:\s*var\(--color-neutral-200\)/,
    );
  }
  assert.match(
    lastRule(css, '.tool-plan [data-state="completed"] > span'),
    /color:\s*var\(--color-neutral-500\)/,
  );
  const highlightedCode = lastRule(css, ".hljs");
  assert.match(highlightedCode, /color:\s*light-dark\(#24292e,\s*#c9d1d9\)/);
  assert.match(highlightedCode, /background:\s*var\(--color-neutral-100\)/);
  assert.match(
    lastRule(css, ".hljs-keyword"),
    /color:\s*light-dark\(#d73a49,\s*#ff7b72\)/,
  );
  assert.match(
    lastRule(css, ".hljs-string"),
    /color:\s*light-dark\(#032f62,\s*#a5d6ff\)/,
  );
}, 15000);
