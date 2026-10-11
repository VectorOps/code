import * as React from "react";

export type Appearance = "system" | "light" | "dark";

export function parseAppearance(value: string | null): Appearance {
  return value === "light" || value === "dark" ? value : "system";
}

function readAppearance(): Appearance {
  try {
    return parseAppearance(localStorage.getItem("runner.appearance"));
  } catch {
    return "system";
  }
}

const AppearanceContext = React.createContext<{
  value: Appearance;
  change: (value: Appearance) => void;
}>({ value: "system", change: () => {} });

export function AppearanceProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [value, setValue] = React.useState(readAppearance);
  React.useLayoutEffect(() => {
    document.documentElement.dataset.appearance = value;
  }, [value]);

  function change(next: Appearance) {
    setValue(next);
    try {
      localStorage.setItem("runner.appearance", next);
    } catch {}
  }

  return (
    <AppearanceContext.Provider value={{ value, change }}>
      {children}
    </AppearanceContext.Provider>
  );
}

export function AppearanceControl() {
  const { value, change } = React.useContext(AppearanceContext);
  return (
    <label className="search-label">
      Appearance
      <select
        value={value}
        onChange={(event) => change(parseAppearance(event.target.value))}
      >
        <option value="system">System</option>
        <option value="light">Light</option>
        <option value="dark">Dark</option>
      </select>
    </label>
  );
}
