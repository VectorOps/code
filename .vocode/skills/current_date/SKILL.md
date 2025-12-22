---
name: current_date
description: Returns the current date and time in a clear, human-readable format. Use this skill whenever the user asks for the current time or needs a timestamp based on the present moment.
metadata:
  author: local-dev
  version: "1.0"
---

# Test time skill

This skill provides the **current wall clock time** on request.

## How to use this skill

- Invoke this skill when the user explicitly asks for the *current time*, *current date and time*, *what time is it*, or needs a *timestamp for now*.
- You should not guess or fabricate times. Always query the actual current time in your environment right before responding.

## Behavior

1. Retrieve the current time from the runtime environment (e.g., system clock or a trusted time API if available).
2. Present the result in a concise, human-friendly format, such as:
   - `2025-01-30 14:23:05 UTC`
   - `2025-01-30 09:23:05-05:00 (local time)`
3. If both local time and UTC are easily available, include both.

## Output guidelines

- Prefer ISO 8601-style formats (YYYY-MM-DD HH:MM:SS with time zone).
- If the environment has a clear local time zone, specify it by offset or name.
- If you are unsure of the local time zone, clearly label the time as UTC.

Examples:

- "The current time is 2025-01-30 14:23:05 UTC."
- "Right now it is 2025-01-30 09:23:05-05:00 in your local time zone."

## Edge cases

- If you cannot reliably access the current time, say so explicitly and avoid providing a guessed value.
- If time synchronization errors are suspected (e.g., obviously incorrect system clock), mention the uncertainty in your response.
