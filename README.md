VectorOps Code
==============

Build software with an agentic coding companion that understands your project and works through changes step by step. VectorOps Code turns development tasks into a configurable graph of decisions, tool calls, and confirmations so you stay in control while moving faster.

What it is
----------
VectorOps Code is a graph-based orchestration layer for AI-assisted coding. Each workflow is a series of connected steps that collaborate with language models, apply precise changes, request approvals, and adapt based on outcomes. It’s designed to feel like a reliable teammate that reasons, edits, and validates work with you.

Highlights
----------
- Agentic coding flows that plan, edit, review, and iterate
- Outcome-driven routing that adapts to success, retries, or errors
- Integrated tooling to apply changes, run commands, and launch nested tasks
- Interactive terminal experience with streaming output and approvals
- Model-agnostic LLM execution with tool-aware prompting and streaming
- Cost and token tracking to keep usage visible

LLM support
-----------
- Choose from many providers with flexible model selection via LiteLLM
- Streamed responses, function/tool calling, and clear outcome tagging
- Preprocessors for file reading and diff/patch-aware prompting
- Usage and cost estimation to stay within budgets

Configurability first
---------------------
VectorOps Code is built to be configured, not hardcoded:
- Define workflows as graphs with clear node types and outcomes
- Customize confirmation and reset policies to fit your review style
- Tune how messages and results flow between steps
- Enable tools per step and adjust auto-approval behavior
- Layer templates and local overrides to reuse patterns across projects
- Call MCP servers

How it works
------------
- You design a workflow as a directed graph of steps
- LLM-backed steps reason about code and choose outcomes
- Tool steps apply changes or perform side effects
- Edges route to the next step based on outcomes and policies
- You approve or revise at key moments, with the ability to rewind and resume

Why graphs
----------
Graphs make agent behavior transparent and dependable. Each node has a purpose, outcomes are explicit, and transitions are controlled. This structure keeps complex tasks understandable, testable, and easy to evolve as your needs grow.

Extensible by design
--------------------
- Add new tools and make them callable from any step
- Introduce new node types and executors as your workflows evolve
- Start nested workflows for multi-stage jobs and subroutines


Reliable and observable
-----------------------
- Strong runtime guarantees for transitions, retries, and history management
- Clear approval flows for final messages and tool calls
- Detailed accounting for token usage and cost
- Backed by automated tests across the core engine and executors
