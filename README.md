# 🐝 AgentHive

> **Autonomous Local Multi-Agent Workspace with Human-in-the-Loop Governance & Persistent Memory**

<br>

**AgentHive** is an enterprise-grade, local-first multi-agent orchestration platform powered by **[LangGraph 1.2.11](https://github.com/langchain-ai/langgraph)**, **[Ollama](https://ollama.com/)**, **Redis**, **ChromaDB**, and **Streamlit**. 

Designed for complete data privacy and offline autonomy, AgentHive coordinates a team of specialized AI agents — **Supervisor, Coder, Data Specialist, Researcher, Writer, and Reviewer** — to break down complex tasks, execute sandboxed code, perform SQL database operations, and retrieve long-term vector memories without relying on third-party cloud APIs.

<br>

## 🎯 KEY HIGHLIGHTS
1. <b>Multi-Agent Orchestration</b>
Supervisor: Intelligently decomposes requests into ordered subtasks, assigning each subtask to the optimal specialist.
Specialists Network:
Research: Factual research and information discovery using DuckDuckGo search (web_search).
Coder: Python code generation, AST-validated sandbox execution (python_sandbox), and workspace file operations (workspace_file).
Data: SQL database query execution (db_query), data analysis, and sandboxed Python statistical calculations (python_sandbox).
Writer: Structured long-form prose, executive summaries, and reports, incorporating web search results (web_search) and external HTTP API data (api_call).
Reviewer: Evaluates specialist outputs against quality criteria, triggering retries or human escalations if confidence is low.
2. <b>Two-Tier Memory System</b>
Short-Term Task Memory (Redis): Scoped per task_id with automatic TTL cleanup for intermediate subtask outputs.
Long-Term Semantic Memory (ChromaDB): Embedded via nomic-embed-text. Stores past outcomes, tools used, domain facts, and user preferences scoped by user_id.
3. <b>Human-in-the-Loop Safety & Checkpointing</b>
LangGraph 1.2.11 SQLite Saver (SqliteSaver): Thread states persist to disk (data/agent_hive.db), surviving Streamlit reruns and process restarts.
Pre-Execution Sensitive Operation Detection: Pauses prior to non-read-only SQL statements (INSERT, UPDATE), non-idempotent HTTP methods (POST, PUT, DELETE), or file-write code operations.
Granular Approvals: Notify, Approve Action, Approve Plan (bypass future step pauses), and Take Over (manual human response override).
4.<b> Trace Explorer & Replay Engine</b>
Unified Execution Trace: Combines LangGraph checkpoint snapshots, tool audit JSONL logs (data/tool_calls.jsonl), and human approval audit logs into a single step-by-step tree.
Replay System:
Full Replay: Re-run tasks from scratch with identical inputs under a new task_id.
Partial Replay: Edit any step's output in the tree and resume execution.
Trace Diff Viewer: Highlights step-by-step matches, divergences, and exact divergence points between original and replay runs.

<br>
<hr>
<br>


### 🛠️ Tech Stack

| Component | Tool / Library | Version | Why This Choice |
| :--- | :--- | :--- | :--- |
| **Language** | Python | `3.13` | Modern typing semantics, speed improvements, and strict async runtime safety. |
| **Orchestration** | LangGraph | `1.2.11` | Native DAG state machine with pre-execution interrupts (`interrupt()`) & Pregel loop controls. |
| **LLM Runtime** | Ollama | `0.6.2` | Local GPU-accelerated LLM inference without cloud API costs or data leakage risks. |
| **Short-Term Memory** | Redis | `>=5.0.0` | Sub-millisecond task-scoped key-value working memory with TTL auto-expiration per `task_id`. |
| **Long-Term Memory** | ChromaDB | `>=0.5.0` | Embedded local vector database for semantic memory retrieval (`nomic-embed-text`). |
| **Checkpointing** | SqliteSaver | `>=3.1.0` | Thread-safe disk-backed checkpointing surviving Streamlit reruns (`langgraph-checkpoint-sqlite`). |
| **UI Framework** | Streamlit | `>=1.37.0` | Rapid local interactive dashboard with custom Sapphire/Spruce CSS visual design. |

<br>
<hr>
<br>


### 🏗️ System Architecture

AgentHive orchestrates autonomous specialists through a stateful graph pipeline with human-in-the-loop safety gates, two-tier memory, and checkpointed traces.

<br>

```text
                     ┌─────────────────────────────────────────┐
                     │          User Request / Task            │
                     └────────────────────┬────────────────────┘
                                          │
                                          ▼
                     ┌─────────────────────────────────────────┐
                     │      Memory Recall Agent (ChromaDB)     │
                     └────────────────────┬────────────────────┘
                                          │
                                          ▼
                     ┌─────────────────────────────────────────┐
                     │       Supervisor Agent (Planner)        │
                     └────────────────────┬────────────────────┘
                                          │
                                          ▼
                     ┌─────────────────────────────────────────┐
                     │    Pre-Execution Approval Check Gate    │
                     │  (Sensitive Op / Low Conf / Escalations)│
                     └──────────┬───────────────────┬──────────┘
                                │                   │
                    [Approved]  ▼                   ▼  [Paused]
    ┌──────────────────────────────────────┐     ┌──────────────────────┐
    │          Specialist Network          │     │  SQLite Approval UI  │
    │ ┌───────────┬─────────┬───────┬────┐ │     │ (Human Review/Resume)│
    │ │ Research  │ Coder   │ Data  │Writer│ │     └──────────┬───────────┘
    │ └───────────┴─────────┴───────┴────┐ │                │
    └───────────────────┬──────────────────┘                │ [Command(resume=...)]
                        │                                   │
                        └───────────────────┬───────────────┘
                                            │
                                            ▼
                     ┌─────────────────────────────────────────┐
                     │            Reviewer Agent               │
                     └────────────────────┬────────────────────┘
                                          │
                                          ▼
                     ┌─────────────────────────────────────────┐
                     │       Memory Save Agent (ChromaDB)      │
                     └────────────────────┬────────────────────┘
                                          │
                                          ▼
                     ┌─────────────────────────────────────────┐
                     │  Unified Execution Trace Explorer UI    │
                     └─────────────────────────────────────────┘
```

<br>
<hr>
<br>

### 📁 Project Structure

```text
src/
├── agents/          # Autonomous Supervisor, Specialist network (Research, Coder, Data, Writer), and Reviewer
├── memory/          # Task-scoped Redis short-term store & ChromaDB long-term semantic vector store
├── orchestration/   # LangGraph DAG topology, SqliteSaver checkpointer, approval queue, & execution trace builder
├── tools/           # Sandboxed tool registry (python_sandbox, db_query, web_search, api_call, workspace_file)
├── llm/             # Application-specific Ollama client & local embedding wrappers
└── ui/              # Custom Streamlit views, layout renderers, and Sapphire/Spruce visual theme
```

<br>
<hr>
<br>


### 📸 UI Screenshots

![Dashboard](docs/screenshots/dashboard.png)

<br>

![Trace Explorer](docs/screenshots/trace_explorer.png)

<br>

![Approval Queue](docs/screenshots/approval_queue.png)

<br>
<hr>
<br>

### 🚀 Quick Start Guide

> 💡 **Which mode should I use?**
> - **Option A (Docker Compose)**: Best for live demos and single-command evaluation. Spins up Streamlit, Redis, and PostgreSQL in containers.
> - **Option B (Native Streamlit Run)**: Recommended for active development with instant hot-reloading (`streamlit run app.py`).

<br>

#### Prerequisites
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

<br>

#### Option A: Docker Compose (Recommended)
```bash
git clone https://github.com/Ishita-rastogi06/AgentHive.git
cd AgentHive
docker-compose up --build
```
Open **`http://localhost:8501`**. Starts Streamlit (8501), Redis (6379), and PostgreSQL (host port `5433`).

<br>

#### Option B: Native Local
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows
source .venv/bin/activate           # Linux/macOS

pip install -r requirements.txt
cp .env.example .env

docker-compose up -d postgres redis # start background services
streamlit run app.py
```

<br>
<hr>
<br>

### 🧪 Automated Test Suite

```bash
py -m pytest
```

```text
================= 240 passed, 8 warnings in 114.91s (0:01:54) =================
```

<br>
<hr>
<br>

### ⚙️ Environment Variables Reference

| Variable | Default | Description |
| :--- | :--- | :--- |
| `OLLAMA_HOST` | `http://localhost:11434` | URL of the local Ollama service endpoint. |
| `OLLAMA_MODEL` | `llama3.2:3b` | Chat model used for all agent planning and execution calls. |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model used by ChromaDB long-term memory. |
| `LLM_TIMEOUT_SECONDS` | `60` | Maximum seconds to wait for a single LLM response before timing out. |
| `LLM_MAX_TOKENS` | `1200` | Maximum number of new tokens generated per LLM response. |
| `LLM_CONTEXT_WINDOW` | `4096` | Context window size (in tokens) passed to Ollama. |
| `LLM_TEMPERATURE` | `0.2` | Sampling temperature (0.0 = deterministic, 1.0 = creative). |
| `ENABLE_OFFLINE_FALLBACK` | `true` | Enables deterministic local fallback messages when Ollama is unreachable. |
| `MAX_RETRIES` | `2` | Maximum retry attempts by reviewer before human escalation. |
| `LOW_CONFIDENCE_THRESHOLD` | `0.60` | Reviewer confidence score threshold triggering retries/escalation. |
| `SANDBOX_TIMEOUT_SECONDS` | `10` | Timeout in seconds for Python sandbox code execution. |
| `API_TIMEOUT_SECONDS` | `15` | Timeout in seconds for outbound HTTP API tool calls. |
| `MAXIMUM_TOOL_OUTPUT_CHARS` | `8000` | Maximum characters captured per tool output before truncation. |
| `ALLOWED_API_DOMAINS` | `api.github.com`,<br>`jsonplaceholder.typicode.com`,<br>`api.duckduckgo.com` | Comma-separated allowlist of hostnames reachable by API tools. |
| `REDIS_URL` | `""` | Connection URL for Redis short-term working memory (falls back to in-memory). |
| `DATA_DIR` | `./data` | Local directory path for persistent SQLite database and audit logs. |
| `TEMP_DIR` | `./temp` | Local directory path for transient scratch files. |
| `WORKSPACE_DIR` | `./workspace` | Restricted workspace directory path for file tools. |

<br>
<hr>
<br>


### 🎯 Design Decisions

- **Local-First LLM Architecture**: Built on Ollama (`llama3.2:3b`) and local vector embeddings (`nomic-embed-text`) for zero cloud API costs and offline capability. Decoupled inside `src/llm/ollama_client.py` for easy provider swapping.

- **Single-Instance Disk Checkpointing (`SqliteSaver`)**: Thread state persists to disk (`data/agent_hive.db`) across Streamlit reruns without requiring Postgres container overhead.

- **Deny-by-Default Outbound Network Security**: The `api_call` tool enforces a strict host allowlist configured via `ALLOWED_API_DOMAINS` to prevent unauthorized outbound HTTP requests.

<br>

#### ⚠️ Known System Constraints & Architectural Boundaries

- **Targeted Deterministic Guards vs. General Fact-Checking**: The Reviewer incorporates targeted python overrides for specific failure modes. It is a safety auditor, not a universal general-purpose fact checker.

- **LLM Factual Grounding Capacity**: Small local models (`llama3.2:3b`) have parameter limits; grounding against `web_search` mitigates hallucinations.

- **Graceful Memory Fallback**: When Redis is absent, short-term memory gracefully falls back to thread-safe in-memory Python dictionaries.

<br>
<hr>
<br>
## 👩‍💻 Developer

**Ishita Rastogi** (B.Tech CSE)  
*GitHub*: [Ishita-rastogi06](https://github.com/Ishita-rastogi06)
