import * as React from "react";

const Context = React.createContext(false);

export const DebugProvider = Context.Provider;
export const useDebug = () => React.useContext(Context);

export function DebugOnly({ children }: { children: () => React.ReactNode }) {
  return useDebug() ? <>{children()}</> : null;
}
