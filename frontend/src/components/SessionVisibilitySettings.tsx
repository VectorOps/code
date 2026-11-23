import { useEffect, useState } from "react";
import { useDesktopSettings } from "./DesktopSettings";

export function SessionVisibilitySettings() {
  const { value, ready, save } = useDesktopSettings();
  const [days, setDays] = useState(String(value.auto_hide_after_days));
  useEffect(
    () => setDays(String(value.auto_hide_after_days)),
    [value.auto_hide_after_days],
  );
  const parsed = Number(days);
  const valid =
    days.trim() !== "" &&
    Number.isInteger(parsed) &&
    parsed >= 0 &&
    parsed <= 36500;
  return (
    <section aria-label="Recent workflows">
      <h4>Recent workflows</h4>
      <label>
        Hide after inactivity (days)
        <input
          type="number"
          min={0}
          max={36500}
          step={1}
          value={days}
          disabled={!ready}
          aria-invalid={!valid}
          onChange={(event) => setDays(event.target.value)}
        />
      </label>
      <button
        type="button"
        disabled={!ready || !valid || parsed === value.auto_hide_after_days}
        onClick={() => save({ auto_hide_after_days: parsed })}
      >
        Save inactivity limit
      </button>
      <p>
        Default: 7 days. Set to 0 to disable automatic hiding. Saved on this
        host and shared across projects.
      </p>
      <p>
        Cleanup runs in the background when the worker starts or you start a
        workflow. Hidden workflows remain available in All. Showing one in
        Recent gives it another full inactivity period before it can be hidden
        automatically again.
      </p>
    </section>
  );
}
