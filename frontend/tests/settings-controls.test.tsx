import * as testing from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import * as vitest from "vitest";
import { SettingField } from "../src/components/SettingsControls";

vitest.test.each([
  { value: "openai/base", numeric: false, multiline: false },
  { value: 0, numeric: true, multiline: false },
  { value: "Follow these instructions", numeric: false, multiline: true },
  { value: "", numeric: false, multiline: true },
])(
  "shows configured value only while editing $value",
  async ({ value, numeric, multiline }) => {
    const user = userEvent.setup();
    const apply = vitest.vi.fn(async () => {});
    const reset = vitest.vi.fn(async () => {});
    const field = { name: "value", label: "Value", numeric, multiline };
    const view = testing.render(
      <SettingField field={field} value={value} apply={apply} reset={reset} />,
    );
    const input = view.getByLabelText("Value");
    vitest.expect(input).toHaveProperty("value", String(value));
    vitest.expect(view.queryByText(/Configured:/)).toBeNull();

    await user.clear(input);
    await user.type(input, numeric ? "1" : "New value");
    vitest
      .expect(view.getByText(/Configured:/).textContent)
      .toBe(`Configured: ${JSON.stringify(value)}`);

    await user.clear(input);
    if (String(value)) await user.type(input, String(value));
    vitest.expect(view.queryByText(/Configured:/)).toBeNull();

    await user.clear(input);
    const next = numeric ? 2 : "Applied value";
    await user.type(input, String(next));
    await user.click(view.getByRole("button", { name: "Apply value" }));
    vitest.expect(apply).toHaveBeenCalledExactlyOnceWith(next, false);
    view.rerender(
      <SettingField field={field} value={next} apply={apply} reset={reset} />,
    );
    vitest.expect(view.queryByText(/Configured:/)).toBeNull();
    vitest.expect(input).toHaveProperty("value", String(next));
    vitest.expect(reset).not.toHaveBeenCalled();
  },
);

vitest.test.each([undefined, null, false, ["invalid"], { invalid: true }])(
  "keeps configured values that the input cannot represent visible: %j",
  async (value) => {
    const user = userEvent.setup();
    const view = testing.render(
      <SettingField
        field={{ name: "value", label: "Value" }}
        value={value}
        apply={vitest.vi.fn(async () => {})}
        reset={vitest.vi.fn(async () => {})}
      />,
    );
    const input = view.getByLabelText("Value");
    vitest.expect(input).toHaveProperty("value", "");
    const configured =
      value === undefined ? "Not configured" : JSON.stringify(value);
    vitest
      .expect(view.getByText(/Configured:/).textContent)
      .toBe(`Configured: ${configured}`);
    await user.type(input, "Draft");
    vitest
      .expect(view.getByText(/Configured:/).textContent)
      .toBe(`Configured: ${configured}`);
  },
);
