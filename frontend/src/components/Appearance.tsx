import * as React from "react";
import { useDesktopSettings } from "./DesktopSettings";

export type Appearance = "system" | "light" | "dark";

export function parseAppearance(value: string | null): Appearance {
  return value === "light" || value === "dark" ? value : "system";
}

export function AppearanceProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const { value } = useDesktopSettings();
  React.useLayoutEffect(() => {
    document.documentElement.dataset.appearance = value.appearance;
  }, [value.appearance]);
  return <>{children}</>;
}

export function AppearanceControl() {
  const { value, ready, save } = useDesktopSettings();
  return (
    <label className="search-label">
      Appearance
      <select
        value={value.appearance}
        disabled={!ready}
        onChange={(event) =>
          save({ appearance: parseAppearance(event.target.value) })
        }
      >
        <option value="system">System</option>
        <option value="light">Light</option>
        <option value="dark">Dark</option>
      </select>
    </label>
  );
}
