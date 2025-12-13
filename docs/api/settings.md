# Settings

VectorOps Code is configured through two layers of settings:

- `vocode.settings.Settings` controls workflows, tools, UI, logging, MCP, and
  process execution for the VectorOps Code runtime.
- `knowlt.settings.ProjectSettings` (embedded as `Settings.know`) controls how
  the underlying code and documentation are scanned, indexed, embedded, and
  searched.

This page documents both models in detail, including all fields and
representative configuration examples.

---

## Vocode settings (`vocode.settings.Settings`)

The top-level configuration object for VectorOps Code is
`vocode.settings.Settings`. It is typically loaded from a YAML file and passed
to the main entrypoints.

### Top-level layout

```yaml
# .vocode.yaml

workflows:
  default:
    description: "Main agentic coding workflow"
    config: {}
    nodes: []
    edges: []

default_workflow: default

tools:
  - name: exec
    enabled: true
    auto_approve: false
    config:
      timeout_s: 120

know:  # See "Knowlt project settings" below
  project_name: "my-service"
  repo_name: "my-service"
  repo_path: "."

ui:
  multiline: true
  edit_mode: emacs
  show_banner: true

logging:
  default_level: info
  enabled_loggers:
    asyncio: debug

mcp:
  servers:
    docs-mcp:
      url: "http://localhost:9000/mcp"
  tools_whitelist:
    - docs.search

process:
  backend: local
  env:
    inherit_parent: true
    allowlist:
      - PATH
      - HOME
    defaults:
      LANG: en_US.UTF-8
  shell:
    type: bash
    program: bash
    args: ["--noprofile", "--norc"]
    default_timeout_s: 120

exec_tool:
  max_output_chars: 16384

tools_runtime:
  max_concurrent: 4

tool_call_formatters:
  exec:
    title: "Shell command"
    formatter: "generic"
    show_output: true
    options:
      truncate_after: 2000
```

### `Settings`

`vocode.settings.Settings` aggregates all runtime knobs used by the agent.

#### Fields

##### `workflows: dict[str, WorkflowConfig]`

Mapping from workflow name to workflow configuration.

Example:

```yaml
workflows:
  default:
    description: "Main coding workflow"
    config:
      confirm_on_apply: true
    nodes: []
    edges: []
```

##### `default_workflow: str | null`

Optional name of the workflow to auto-start in interactive UIs.

Example:

```yaml
default_workflow: default
```

##### `tools: list[ToolSpec]`

Global tool specifications applied across workflows.

Example:

```yaml
tools:
  - name: exec
    enabled: true
    auto_approve: false
  - name: apply_patch
    enabled: true
    auto_approve: true
```

##### `know: knowlt.settings.ProjectSettings | null`

Optional embedded Knowlt project configuration for scanning and indexing.

Example (high level, see detailed section later):

```yaml
know:
  project_name: "my-service"
  repo_path: "."
  embedding:
    enabled: true
    cache_backend: duckdb
```

##### `ui: UISettings | null`

Optional terminal UI behavior configuration.

Example:

```yaml
ui:
  multiline: true
  edit_mode: vim
  show_banner: false
```

##### `logging: LoggingSettings | null`

Optional logging configuration.

Example:

```yaml
logging:
  default_level: info
  enabled_loggers:
    vocode.runner: debug
    asyncio: warning
```

##### `mcp: MCPSettings | null`

Optional Model Context Protocol configuration.

Example:

```yaml
mcp:
  servers:
    local-tools:
      command: "uv"
      args: ["run", "my-mcp-server"]
      env:
        MY_ENV: value
  tools_whitelist:
    - filesystem.read
    - filesystem.write
```

##### `process: ProcessSettings | null`

Optional process subsystem configuration, including shell and environment.

Example:

```yaml
process:
  backend: local
  env:
    inherit_parent: true
    allowlist: ["PATH", "HOME"]
    denylist: ["AWS_SECRET_ACCESS_KEY"]
    defaults:
      LANG: en_US.UTF-8
  shell:
    type: bash
    program: bash
    args: ["--noprofile", "--norc"]
    default_timeout_s: 300
```

##### `exec_tool: ExecToolSettings | null`

Optional global limits for the `exec` tool.

Example:

```yaml
exec_tool:
  max_output_chars: 20000
```

##### `tools_runtime: ToolRuntimeSettings | null`

Optional concurrency limits for tool execution.

Example:

```yaml
tools_runtime:
  max_concurrent: 3
```

##### `tool_call_formatters: dict[str, ToolCallFormatter]`

Per-tool formatting configuration for the terminal UI.

Example:

```yaml
tool_call_formatters:
  exec:
    title: "Exec command"
    formatter: "generic"
    show_output: true
    options:
      truncate_after: 4000
  apply_patch:
    title: "Apply patch"
    formatter: "diff"
    show_output: false
```

#### `WorkflowConfig`

Defines a single workflow used by the agent.

##### `name: str | null`

Name of the workflow. It is normally populated automatically from the key in
`workflows`, so you rarely need to set it explicitly in YAML.

##### `description: str | null`

Human-readable purpose for the workflow.

Example:

```yaml
workflows:
  default:
    description: "End-to-end coding assistant workflow"
```

##### `config: dict[str, Any]`

Free-form workflow-level configuration, passed to nodes and runners.

Example:

```yaml
workflows:
  default:
    config:
      confirm_on_large_diff: true
      max_retries: 3
```

##### `nodes: list[vocode.models.Node]`

Node definitions for the workflow graph.

Example (simplified):

```yaml
workflows:
  default:
    nodes:
      - kind: llm
        id: plan
        config:
          prompt: "Plan the changes."
      - kind: apply_patch
        id: apply
```

##### `edges: list[vocode.models.Edge]`

Directed edges linking nodes based on outcomes.

Example:

```yaml
workflows:
  default:
    edges:
      - source: plan
        outcome: success
        target: apply
```

#### `ToolSpec`

Controls how a tool is exposed globally or in a workflow.

##### `name: str`

Tool name as registered in the tool registry (for example, `exec` or
`apply_patch`).

Example:

```yaml
tools:
  - name: exec
```

##### `enabled: bool`

Whether the tool is globally available. Defaults to `true`.

Example:

```yaml
tools:
  - name: exec
    enabled: false
```

##### `auto_approve: bool | null`

If `true`, tool calls can be auto-approved in the UI when allowed by
`auto_approve_rules` and UI policy.

Example:

```yaml
tools:
  - name: apply_patch
    auto_approve: true
```

##### `auto_approve_rules: list[ToolAutoApproveRule]`

Optional rules that allow auto-approval for certain argument patterns.

Example:

```yaml
tools:
  - name: exec
    auto_approve: false
    auto_approve_rules:
      - key: "command"
        pattern: "^pytest( |$)"
```

##### `config: dict[str, Any]`

Free-form tool-specific configuration passed to the implementation.

Example:

```yaml
tools:
  - name: exec
    config:
      timeout_s: 180
      working_dir: "."
```

#### `ToolAutoApproveRule`

Defines a single auto-approval rule on JSON tool arguments.

##### `key: str`

Dot-separated path inside the tool arguments (for example `resource.action` or
`command`).

##### `pattern: str`

Regular expression string applied to the stringified value at `key`.

Example:

```yaml
auto_approve_rules:
  - key: "resource.action"
    pattern: "^(read|list)_files$"
```

#### `ToolCallFormatter`

Configures how a tool call is displayed in the terminal UI.

##### `title: str`

Display name for the tool call.

Example values: `"Shell command"`, `"Apply patch"`.

##### `formatter: str`

Formatter implementation name (for example `"generic"` or `"diff"`).

Example:

```yaml
tool_call_formatters:
  exec:
    formatter: "generic"
```

##### `show_output: bool`

If `true`, show tool output by default instead of requiring a manual
expansion.

Example:

```yaml
tool_call_formatters:
  exec:
    show_output: true
```

##### `options: dict[str, Any]`

Free-form options for the chosen formatter.

Example:

```yaml
tool_call_formatters:
  exec:
    options:
      truncate_after: 2000
```

#### `UISettings`

Controls interactive terminal UI behavior.

##### `multiline: bool`

If `true`, input accepts multiple lines; Enter inserts a newline.

Example:

```yaml
ui:
  multiline: true
```

##### `edit_mode: "emacs" | "vim" | null`

Optional editing mode override; `null` uses the library default (`emacs`).

Example:

```yaml
ui:
  edit_mode: vim
```

##### `show_banner: bool`

Whether to display the startup banner.

Example:

```yaml
ui:
  show_banner: false
```

#### `LoggingSettings`

Configures logging defaults and per-logger overrides.

##### `default_level: vocode.state.LogLevel`

Default log level for main loggers (`vocode`, `knowlt`).

Example:

```yaml
logging:
  default_level: info
```

##### `enabled_loggers: dict[str, LogLevel]`

Mapping from logger name to specific log level.

Example:

```yaml
logging:
  enabled_loggers:
    vocode.runner: debug
    asyncio: warning
```

#### `MCPServerSettings` and `MCPSettings`

`MCPServerSettings` describes a single MCP server; `MCPSettings` is the
top-level MCP configuration.

##### `MCPServerSettings.url: str | null`

HTTP URL for a FastMCP-compatible server.

Example:

```yaml
mcp:
  servers:
    docs:
      url: "http://localhost:9000/mcp"
```

##### `MCPServerSettings.command: str | null`

Local command to start a server instead of `url`.

Example:

```yaml
mcp:
  servers:
    tools:
      command: "uv"
      args: ["run", "my_mcp_server"]
```

##### `MCPServerSettings.args: list[str]`

Command-line arguments passed to `command`.

##### `MCPServerSettings.env: dict[str, str]`

Additional environment variables for the MCP server.

##### `MCPSettings.servers: dict[str, MCPServerSettings]`

Mapping of server name to configuration.

##### `MCPSettings.tools_whitelist: list[str] | null`

Optional allowlist of tool names exposed by discovered MCP servers.

Example:

```yaml
mcp:
  tools_whitelist:
    - search.files
    - search.docs
```

#### `ProcessEnvSettings`, `ShellSettings`, and `ProcessSettings`

These settings control how subprocesses and shells are spawned.

##### `ProcessEnvSettings.inherit_parent: bool`

If `true`, inherit environment from the parent process.

##### `ProcessEnvSettings.allowlist: list[str] | null`

If set, only these environment variables are passed through.

##### `ProcessEnvSettings.denylist: list[str] | null`

Environment variables to explicitly strip out.

##### `ProcessEnvSettings.defaults: dict[str, str]`

Default environment variables to always set.

Example:

```yaml
process:
  env:
    inherit_parent: true
    allowlist: ["PATH", "HOME"]
    denylist: ["AWS_SECRET_ACCESS_KEY"]
    defaults:
      LANG: en_US.UTF-8
```

##### `ShellSettings.type: "bash"`

Shell type (currently `bash`).

##### `ShellSettings.program: str`

Shell executable.

##### `ShellSettings.args: list[str]`

Arguments used when starting the shell.

##### `ShellSettings.default_timeout_s: int`

Default per-command timeout in seconds.

Example:

```yaml
process:
  shell:
    type: bash
    program: bash
    args: ["--noprofile", "--norc"]
    default_timeout_s: 300
```

##### `ProcessSettings.backend: str`

Process backend key (commonly `local`).

##### `ProcessSettings.env: ProcessEnvSettings`

Process-level environment configuration.

##### `ProcessSettings.shell: ShellSettings`

Shell configuration used by the process backend.

#### `ExecToolSettings`

Controls limits for the `exec` tool.

##### `max_output_chars: int`

Maximum combined stdout/stderr characters the tool will return. Guards against
runaway output.

Example:

```yaml
exec_tool:
  max_output_chars: 20000
```

#### `ToolRuntimeSettings`

Controls runtime behavior for tool execution.

##### `max_concurrent: int | null`

Maximum number of tool calls to execute concurrently for a single
interaction. `null` or `<= 0` means unlimited.

Example:

```yaml
tools_runtime:
  max_concurrent: 3
```

---

## Knowlt project settings (`knowlt.settings.ProjectSettings`)

`ProjectSettings` configures how the underlying codebase and documentation are
scanned, embedded, and searched. VectorOps Code consumes these settings via
`Settings.know`.

### Typical YAML configuration

```yaml
know:
  project_name: "my-service"
  repo_name: "my-service"
  repo_path: "."

  repository_backend: duckdb
  repository_connection: "./.cache/my-service.duckdb"

  scanner_num_workers: 8
  ignored_dirs:
    - .git
    - .venv
    - node_modules

  embedding:
    enabled: true
    calculator_type: local
    model_name: all-MiniLM-L6-v2
    device: cuda
    batch_size: 256
    sync_embeddings: true
    cache_backend: duckdb
    cache_path: "./.cache/embeddings.duckdb"
    cache_size: 500000
    cache_trim_batch_size: 1000

  chunking:
    chunker_type: recursive
    max_tokens: 512
    min_tokens: 64

  tokenizer:
    default: code

  tools:
    disabled: []
    outputs:
      list_files: json
    file_list_limit: 100

  refresh:
    enabled: true
    cooldown_minutes: 5
    refresh_all_repos: false

  search:
    default_repo_boost: 1.2
    rrf_k: 60
    rrf_code_weight: 0.5
    rrf_fts_weight: 0.5
    embedding_similarity_threshold: 0.4
    bm25_score_threshold: 0.1

  paths:
    enable_project_paths: true

  languages:
    python:
      extra_extensions: [".pyi"]
      venv_dirs: [".venv", "venv"]
      module_suffixes: [".py", ".pyc"]
    text:
      extra_extensions: [".md", ".rst"]
```

### `ProjectSettings`

Top-level project configuration used by Knowlt.

#### Fields

##### `project_name: str | null`

Human-readable project name.

Example: `"my-service"`.

##### `repo_name: str | null`

Repository name (used in multi-repo setups and project paths).

Example: `"my-service"`.

##### `repo_path: str | null`

Filesystem path to the project root to be scanned.

Example values include `"."` and `"../backend"`.

##### `repository_backend: str | null`

Backend used for storing metadata, typically `duckdb`.

Example:

```yaml
repository_backend: duckdb
```

##### `repository_connection: str | null`

Connection string or path for the selected backend.

Example:

```yaml
repository_connection: "./.cache/my-service.duckdb"
```

##### `scanner_num_workers: int | null`

Number of worker threads used by the scanner; `null` auto-selects based on CPU.

Example:

```yaml
scanner_num_workers: 8
```

##### `ignored_dirs: set[str]`

Directory names to skip during scanning.

Example:

```yaml
ignored_dirs:
  - .git
  - .venv
  - node_modules
```

##### `embedding: EmbeddingSettings`

Embedding-specific configuration (see below).

##### `chunking: ChunkingSettings`

Text chunking configuration.

##### `tokenizer: TokenizerSettings`

Default tokenizer selection.

##### `tools: ToolSettings`

Per-tool configuration for Knowlt tools.

##### `refresh: RefreshSettings`

Automatic refresh behavior for project data.

##### `search: SearchSettings`

Search ranking and thresholds.

##### `paths: PathsSettings`

Virtual path behavior.

##### `languages: dict[str, LanguageSettings]`

Language-specific parsing and indexing options.

### `EmbeddingSettings`

Controls how embeddings are computed and cached.

##### `calculator_type: str`

Embedding calculator implementation (for example `"local"`).

Example: `calculator_type: local`.

##### `model_name: str`

Sentence-transformer model name or local path.

Example: `model_name: all-MiniLM-L6-v2`.

##### `device: str | null`

Torch device to use (`"cpu"`, `"cuda"`, etc.). If `null`, a suitable device
is chosen automatically.

Example: `device: cuda`.

##### `batch_size: int`

Batch size for embedding inference.

Example: `batch_size: 256`.

##### `enabled: bool`

Enables embeddings and semantic search tools.

Example: `enabled: true`.

##### `sync_embeddings: bool`

Whether to keep embeddings synchronized with the latest scan.

Example: `sync_embeddings: true`.

##### `cache_path: str | null`

Path or connection string for the embedding cache backend.

Example: `cache_path: "./.cache/embeddings.duckdb"`.

##### `cache_backend: str`

Cache backend: `"duckdb"`, `"sqlite"`, or `"none"`.

Example: `cache_backend: duckdb`.

##### `cache_size: int | null`

Maximum number of cache records (LRU); `null` means unlimited.

Example: `cache_size: 500000`.

##### `cache_trim_batch_size: int`

Number of records to delete when trimming.

Example: `cache_trim_batch_size: 1000`.

### `ToolSettings` (Knowlt)

##### `disabled: set[str]`

Tool names to disable.

Example:

```yaml
tools:
  disabled:
    - search.symbols
```

##### `outputs: dict[str, ToolOutput]`

Per-tool output format overrides (`"json"` or `"structured_text"`).

Example:

```yaml
tools:
  outputs:
    list_files: json
    search_code: structured_text
```

##### `file_list_limit: int`

Default maximum number of files returned by file-listing tools.

Example: `file_list_limit: 100`.

### `RefreshSettings` (Knowlt)

##### `enabled: bool`

Whether to auto-refresh project data periodically.

Example: `enabled: true`.

##### `cooldown_minutes: int`

Minimum time between refreshes in minutes.

Example: `cooldown_minutes: 5`.

##### `refresh_all_repos: bool`

If `true`, refresh all associated repositories instead of just the primary
repository.

Example: `refresh_all_repos: false`.

### `SearchSettings`

Controls multi-signal ranking and thresholds.

##### `default_repo_boost: float`

Boost factor for results from the default repo when a free-text query is
provided. Values greater than `1` boost, values less than `1` penalize.

Example: `default_repo_boost: 1.2`.

##### `rrf_k: int`

Reciprocal Rank Fusion parameter `k`.

Example: `rrf_k: 60`.

##### `rrf_code_weight: float`

Weight for code embedding similarity scores.

Example: `rrf_code_weight: 0.5`.

##### `rrf_fts_weight: float`

Weight for full-text search scores.

Example: `rrf_fts_weight: 0.5`.

##### `embedding_similarity_threshold: float`

Minimum cosine similarity for a result to be accepted.

Example: `embedding_similarity_threshold: 0.45`.

##### `bm25_score_threshold: float | null`

Optional minimum BM25 score for text search results.

Example: `bm25_score_threshold: 0.1`.

##### `node_kind_boosts: dict[NodeKind, float]`

Per-node-kind boost factors (for example, functions vs variables).

Example:

```yaml
search:
  node_kind_boosts:
    FUNCTION: 2.0
    CLASS: 1.5
    VARIABLE: 1.2
```

##### `fts_field_boosts: dict[str, int]`

Integer boosts for fields in full-text search (for example `name` and
`docstring`).

Example:

```yaml
search:
  fts_field_boosts:
    name: 3
    file_path: 2
    body: 1
```

### `PathsSettings`

##### `enable_project_paths: bool`

When `true`, virtual paths are prefixed with the repository name, which is
useful in multi-repo setups.

Example: `enable_project_paths: true`.

### `TokenizerSettings` and `TokenizerType`

##### `TokenizerSettings.default: TokenizerType`

Default tokenizer type: `"noop"`, `"code"`, or `"word"`.

Example:

```yaml
tokenizer:
  default: code
```

### `ChunkingSettings`

##### `chunker_type: str`

Chunker implementation for plain text (for example `"recursive"`).

Example: `chunker_type: recursive`.

##### `max_tokens: int`

Maximum tokens per chunk when embeddings are disabled.

Example: `max_tokens: 512`.

##### `min_tokens: int`

Minimum tokens per chunk before merging.

Example: `min_tokens: 64`.

### Language settings

Language-specific configuration is stored under `languages`.

#### `LanguageSettings`

##### `extra_extensions: list[str]`

Extra file extensions mapped to this language.

Example:

```yaml
languages:
  text:
    extra_extensions: [".md", ".rst"]
```

#### `PythonSettings`

Extends `LanguageSettings` with Python-specific options.

##### `venv_dirs: set[str]`

Directory names treated as virtual environments and typically ignored by the
scanner.

Example:

```yaml
languages:
  python:
    venv_dirs: [".venv", "venv", "env"]
```

##### `module_suffixes: tuple[str, ...]`

File suffixes considered Python modules.

Example:

```yaml
languages:
  python:
    module_suffixes: [".py", ".pyc"]
```

#### `TextSettings`

Uses `LanguageSettings` directly; only `extra_extensions` applies.

---

With these settings, you can fully configure both VectorOps Code's
agentic runtime and the underlying Knowlt-powered project index. For more
details and examples, see the repository at
https://github.com/vectorops/code/.
