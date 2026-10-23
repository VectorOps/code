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
  const feedItem = lastRule(css, ".feed-item");
  assert.match(feedItem, /content-visibility:\s*visible/);
  assert.doesNotMatch(feedItem, /contain-intrinsic-size:/);
  const historyDialog = lastRule(css, ".prompt-history-dialog");
  assert.match(
    historyDialog,
    /height:\s*min\(36rem,\s*(?:calc\()?100dvh\s*-\s*2rem/,
  );
  assert.match(historyDialog, /overflow:\s*hidden/);
  assert.match(historyDialog, /color:\s*var\(--color-neutral-950(?:,|\))/);
  assert.doesNotMatch(
    lastRule(css, ".prompt-history-overlay"),
    /--color-black/,
  );
  assert.match(lastRule(css, ".prompt-history-controls label"), /margin:\s*0/);
  assert.match(lastRule(css, ".prompt-history-controls input"), /height:/);
  const historyResults = lastRule(css, ".prompt-history-results");
  assert.match(historyResults, /flex:\s*1/);
  assert.match(historyResults, /overflow-y:\s*auto/);
  const historyRow = lastRule(css, ".prompt-history-results button");
  assert.match(historyRow, /flex-shrink:\s*0/);
  assert.match(historyRow, /color:\s*var\(--color-neutral-800(?:,|\))/);
  assert.match(historyRow, /background-color:\s*(?:transparent|#0000)/);
  for (const selector of [
    '.prompt-history-results button[aria-selected="true"]',
    ".prompt-history-results button:hover:not(:disabled)",
    ".prompt-history-results button:focus-visible",
    '.prompt-history-results button[aria-selected="true"]:hover:not(:disabled)',
    '.prompt-history-results button[aria-selected="true"]:focus-visible',
  ])
    assert.match(
      lastRule(css, selector),
      /color:\s*var\(--color-neutral-950(?:,|\))/,
    );
  for (const selector of [
    ".prompt-history-results button > span",
    ".prompt-history-project",
  ]) {
    assert.match(lastRule(css, selector), /white-space:\s*nowrap/);
    assert.match(lastRule(css, selector), /text-overflow:\s*ellipsis/);
  }
  assert.match(lastRule(css, ".prompt-history-results small"), /margin:\s*0/);
  assert.match(lastRule(css, ".prompt-history-footer p"), /margin:\s*0/);
  assert.match(lastRule(css, ".composer-help"), /flex:\s*1/);
  assert.match(lastRule(css, ".composer-usage"), /margin-left:\s*auto/);
  assert.match(lastRule(css, ".composer-model"), /border-right/);
  assert.match(lastRule(css, ".composer-cost"), /border-left/);
  const composerButton = lastRule(css, ".composer-submit");
  assert.match(
    composerButton,
    /width:\s*calc\(var\(--spacing(?:,[^)]*)?\)\s*\*\s*8\)/,
  );
  assert.match(
    composerButton,
    /height:\s*calc\(var\(--spacing(?:,[^)]*)?\)\s*\*\s*8\)/,
  );
  assert.match(composerButton, /flex-shrink:\s*0/);
  assert.match(css, /\.image-thumbnail/);
  assert.match(css, /\.tool-plan/);
  const inlinePlan = lastRule(css, ".tool-activity-row > .tool-plan");
  assert.match(inlinePlan, /flex:\s*1/);
  assert.match(inlinePlan, /margin:\s*0/);
  assert.doesNotMatch(inlinePlan, /order:|flex-basis:/);
  assert.match(
    lastRule(css, ".tool-activity-row:has(> .tool-plan)"),
    /align-items:\s*flex-start/,
  );
  assert.match(
    lastRule(css, ".tool-activity-row:has(> .tool-plan-preview)"),
    /align-items:\s*flex-start/,
  );
  const planPreview = lastRule(css, ".tool-plan-preview");
  assert.match(planPreview, /flex:\s*1/);
  assert.match(planPreview, /min-width:\s*0/);
  assert.match(
    lastRule(css, ".tool-plan-notice"),
    /color:\s*var\(--color-neutral-500\)/,
  );
  assert.match(
    lastRule(css, ".tool-group-heading > .tool-plan"),
    /flex-basis:\s*100%/,
  );
  const iconButton = lastRule(css, ".step-details-button");
  assert.match(iconButton, /display:\s*flex/);
  assert.match(iconButton, /color:\s*var\(--color-neutral-400(?:,|\))/);
  assert.match(iconButton, /background-color:\s*(?:transparent|#0000)/);
  assert.match(iconButton, /padding:/);
  assert.match(
    lastRule(css, ".step-card-details"),
    /clip-path:\s*inset\(50%\)/,
  );
  const focusedDetails = lastRule(css, ".step-card-details:focus-visible");
  assert.match(focusedDetails, /clip-path:\s*none/);
  assert.match(focusedDetails, /position:\s*absolute/);
  assert.match(focusedDetails, /width:\s*auto/);
  for (const selector of [
    ".step-details-button:hover:not(:disabled)",
    ".step-details-button:focus-visible",
  ]) {
    assert.match(
      lastRule(css, selector),
      /color:\s*var\(--color-neutral-700(?:,|\))/,
    );
    assert.match(
      lastRule(css, selector),
      /background-color:\s*var\(--color-neutral-100(?:,|\))/,
    );
  }
  assert.match(lastRule(css, ".assistant-controls"), /float:\s*right/);
  assert.match(
    lastRule(css, ".tool-activity-row > .tool-step-controls"),
    /margin-left:\s*auto/,
  );
  const failureRule = lastRule(
    css,
    '.tool-step .entry-state > [aria-label="Failed"]',
  );
  assert.match(failureRule, /color:\s*var\(--color-neutral-400(?:,|\))/);
  assert.doesNotMatch(failureRule, /#a16207|#fbbf24/);
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
