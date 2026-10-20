export type HistoryLimits = {
  initial: number;
  maximum: number;
  target: number;
  older: number;
};

export const defaultHistoryLimits: HistoryLimits = {
  initial: 1000,
  maximum: 2000,
  target: 1000,
  older: 200,
};

export function historyLimits(value: unknown): HistoryLimits {
  if (!value || typeof value !== "object")
    throw new Error("History limits must be an object.");
  const fields = value as Record<string, unknown>;
  const result = { ...defaultHistoryLimits };
  for (const key of Object.keys(result) as (keyof HistoryLimits)[]) {
    const count = fields[key];
    if (typeof count !== "number" || !Number.isSafeInteger(count) || count < 1)
      throw new Error("History counts must be positive whole numbers.");
    result[key] = count;
  }
  if (result.older > 500)
    throw new Error("History requests cannot exceed 500 steps per page.");
  if (result.initial > result.maximum || result.target > result.maximum)
    throw new Error(
      "The retention ceiling must cover the initial window and trim target.",
    );
  return result;
}
