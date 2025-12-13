# Workflows and graphs

VectorOps Code represents each agentic flow as a directed graph. Nodes model
steps such as LLM-backed decisions, tool invocations, and confirmation points,
while edges route between nodes based on outcomes.

At a high level:

- You define workflows as graphs in YAML configuration.
- Each node has a clear type and purpose.
- Edges describe how control moves between nodes for different outcomes
  (success, error, retry, and more).

This graph model makes behavior explicit and testable, and it keeps complex
agent behavior understandable as workflows evolve.
