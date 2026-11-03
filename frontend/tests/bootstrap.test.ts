import { expect, test } from "vitest";
import { readBootstrap } from "../src/communication/bootstrap";

test.each(["1", "0"])(
  "native bootstrap preserves managed mode %s without persisting it",
  (managed) => {
    history.replaceState(
      null,
      "",
      `/#endpoint=ws%3A%2F%2F127.0.0.1%3A1234&token=test&managed=${managed}`,
    );
    expect(readBootstrap()).toEqual({
      endpoint: "ws://127.0.0.1:1234",
      token: "test",
      managed: managed === "1",
    });
    expect(location.hash).toBe("");
  },
);

test("browser development bootstrap does not imply managed ownership or debug mode", () => {
  const element = document.createElement("script");
  element.type = "application/json";
  element.id = "runner-bootstrap";
  element.textContent = JSON.stringify({
    endpoint: "ws://127.0.0.1:8765",
    token: "test",
  });
  document.body.append(element);
  expect(readBootstrap()).toEqual({
    endpoint: "ws://127.0.0.1:8765",
    token: "test",
    managed: false,
  });
  expect(document.getElementById("runner-bootstrap")).toBeNull();
});
