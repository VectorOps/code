export type Json =
  null | boolean | number | string | Json[] | { [key: string]: Json };
export type Params = Record<string, Json>;

export function object(value: Json | undefined): Params {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value
    : {};
}

export const text = (value: Json | undefined) =>
  typeof value === "string" ? value : "";
