# 🐝 AgentHive

**AgentHive** is a production-grade, local-first multi-agent workspace powered by [LangGraph 1.2.11](https://github.com/langchain-ai/langgraph), [Ollama](https://ollama.com/), Redis, ChromaDB, and Streamlit.

---

## 🛠️ Tech Stack

| Component | Tool / Library | Version | Why This Choice |
| :--- | :--- | :--- | :--- |
| **Language** | Python | `3.13` | Modern typing semantics, speed improvements, and strict async runtime safety. |
| **Orchestration** | LangGraph | `1.2.11` | Native DAG state machine support with pre-execution interrupts (`interrupt()`) and Pregel loop controls. |
| **LLM Runtime** | Ollama | `0.6.2` | Local GPU-accelerated LLM inference without cloud API costs or data leakage risks. |
| **Short-Term Memory** | Redis | `>=5.0.0` | Sub-millisecond task-scoped key-value working memory with TTL auto-expiration per `task_id`. |
| **Long-Term Memory** | ChromaDB | `>=0.5.0` | Embedded local vector database for semantic memory retrieval (`nomic-embed-text`) without dedicated server overhead. |
| **Checkpointing** | SqliteSaver (`langgraph-checkpoint-sqlite`) | `>=3.1.0` | Thread-safe disk-backed checkpointing surviving Streamlit reruns without requiring Postgres container overhead. |
| **UI Framework** | Streamlit | `>=1.37.0` | Rapid local interactive dashboard rendering customized with strict Sapphire/Spruce CSS visual design. |

## 🎯 Design Decisions

- **Local-First LLM Architecture**: AgentHive is intentionally built on Ollama (`llama3.2:3b`) and local vector embeddings (`nomic-embed-text`) to guarantee zero cloud API costs, complete privacy, and offline capability. The LLM access layer is decoupled inside `src/llm/ollama_client.py`; swapping in a cloud provider like OpenAI or Claude simply requires writing an adapter class following the same client contract, making provider choice an intentional extension point rather than an architectural lock-in.
- **Single-Instance Disk Checkpointing (`SqliteSaver`)**: Thread state checkpointing uses LangGraph's disk-backed `SqliteSaver` (`data/agent_hive.db`) to provide reliable state persistence across Streamlit reruns and process restarts without forcing users to manage heavy database infrastructure. If horizontal scaling across multiple application nodes is ever required, LangGraph exposes a drop-in Postgres-backed equivalent (`AsyncPostgresSaver` / `PostgresSaver` from `langgraph-checkpoint-postgres`) that uses identical state schemas.
- **Deny-by-Default Outbound Network Security**: The `api_call` tool enforces a strict host allowlist configured via `ALLOWED_API_DOMAINS` to prevent autonomous agents from making unauthorized outbound HTTP requests or data exfiltration calls. This is a deliberate security control designed to enforce the principle of least privilege, requiring explicit domain additions rather than unrestricted web access.

### ⚠️ Known System Constraints & Architectural Boundaries

- **Targeted Deterministic Guards vs. General Fact-Checking**: The Reviewer agent incorporates targeted, deterministic python overrides for specific known failure modes (e.g. mandatory SQL date truncation for monthly queries, and specific known acronym hallucinations like `FAME`/`Faradere`). For general-purpose output, the Reviewer relies on prompt instructions (Rule 6: Web-Search Grounding). It is a targeted requirement auditor and safety guard, **not a universal, general-purpose fact-checking engine** for arbitrary external facts or figures.
- **LLM Factual Grounding Capacity**: Small local models (`llama3.2:3b`) perform excellent local reasoning, code generation, and structured outputs, but open-ended factual knowledge across arbitrary external domains remains bounded by parameter capacity. Grounding against `web_search` tool results mitigates domain hallucinations, but open-ended fact verification remains an inherent model capacity boundary.
- **Graceful Memory Fallback**: When optional infrastructure like Redis is not installed, short-term working memory gracefully falls back to thread-safe in-memory Python dictionaries without throwing runtime exceptions.

---

## 🏗️ System Architecture

AgentHive orchestrates autonomous specialists through a stateful graph pipeline with human-in-the-loop safety gates, persistent two-tier memory, and checkpointed execution traces.

```
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
    │ └───────────┴─────────┴───────┴────┘ │                │
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

---

## 📁 Project Structure

```text
src/
├── agents/          # Autonomous Supervisor, Specialist network (Research, Coder, Data, Writer), and Reviewer
├── memory/          # Task-scoped Redis short-term store & ChromaDB long-term semantic vector store
├── orchestration/   # LangGraph DAG topology, SqliteSaver checkpointer, approval queue, & execution trace builder
├── tools/           # Sandboxed tool registry (python_sandbox, db_query, web_search, api_call, workspace_file)
├── llm/             # Application-specific Ollama client & local embedding wrappers
└── ui/              # Custom Streamlit views, layout renderers, and Sapphire/Spruce visual theme
```

---

## ✨ Key System Capabilities

### 1. Multi-Agent Orchestration
- **Supervisor**: Intelligently decomposes requests into ordered subtasks, assigning each subtask to the optimal specialist.
- **Specialists Network**:
  - **Research**: Factual research and information discovery using DuckDuckGo search (`web_search`).
  - **Coder**: Python code generation, AST-validated sandbox execution (`python_sandbox`), and workspace file operations (`workspace_file`).
  - **Data**: SQL database query execution (`db_query`), data analysis, and sandboxed Python statistical calculations (`python_sandbox`).
  - **Writer**: Structured long-form prose, executive summaries, and reports, incorporating web search results (`web_search`) and external HTTP API data (`api_call`).
- **Reviewer**: Evaluates specialist outputs against quality criteria, triggering retries or human escalations if confidence is low.

### 2. Two-Tier Memory System
- **Short-Term Task Memory (Redis)**: Scoped per `task_id` with automatic TTL cleanup for intermediate subtask outputs.
- **Long-Term Semantic Memory (ChromaDB)**: Embedded via `nomic-embed-text`. Stores past outcomes, tools used, domain facts, and user preferences scoped by `user_id`.

### 3. Human-in-the-Loop Safety & Checkpointing
- **LangGraph 1.2.11 SQLite Saver (`SqliteSaver`)**: Thread states persist to disk (`data/agent_hive.db`), surviving Streamlit reruns and process restarts.
- **Pre-Execution Sensitive Operation Detection**: Pauses prior to non-read-only SQL statements (`INSERT`, `UPDATE`), non-idempotent HTTP methods (`POST`, `PUT`, `DELETE`), or file-write code operations.
- **Granular Approvals**: `Notify`, `Approve Action`, `Approve Plan` (bypass future step pauses), and `Take Over` (manual human response override).

### 4. Trace Explorer & Replay Engine
- **Unified Execution Trace**: Combines LangGraph checkpoint snapshots, tool audit JSONL logs (`data/tool_calls.jsonl`), and human approval audit logs into a single step-by-step tree.
- **Replay System**:
  - **Full Replay**: Re-run tasks from scratch with identical inputs under a new `task_id`.
  - **Partial Replay**: Edit any step's output in the tree and resume execution.
  - **Trace Diff Viewer**: Highlights step-by-step matches, divergences, and exact divergence points between original and replay runs.

---

## ⚙️ Environment Variables Reference

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
| `ALLOWED_API_DOMAINS` | `api.github.com,jsonplaceholder.typicode.com,api.duckduckgo.com` | Comma-separated allowlist of hostnames reachable by API tools. |
| `REDIS_URL` | `""` | Connection URL for Redis short-term working memory. When unset, falls back to in-memory short-term storage. |
| `DATA_DIR` | `./data` | Local directory path for persistent SQLite database and audit logs. |
| `TEMP_DIR` | `./temp` | Local directory path for transient scratch files. |
| `WORKSPACE_DIR` | `./workspace` | Restricted workspace directory path for file tools. |

---

## 📸 UI Screenshots

<!-- Save screenshot files to docs/screenshots/ before viewing -->
![Dashboard](docs/screenshots/dashboard.png)
![Trace Explorer](docs/screenshots/trace_explorer.png)
![Approval Queue](docs/screenshots/approval_queue.png)

---

## 🚀 Quick Start Guide

> 💡 **Which mode should I use?**
> - **Option A (Docker Compose)**: Best for live demos, portfolio walkthroughs, and clean single-command evaluation. Spins up the entire stack (Streamlit, Redis, PostgreSQL) in containerized isolation.
> - **Option B (Native Streamlit Run)**: Recommended for active code development. Allows instant hot-reloading (`streamlit run app.py`) without rebuilding Docker images. Note that Option B coexists with Docker: the app connects to the background PostgreSQL (`localhost:5433`) and Redis (`localhost:6379`) containers started via `docker-compose up -d postgres redis`.

### Prerequisites (Host Machine)
Install [Ollama](https://ollama.com/) on the host machine and pull the required LLM and embedding models:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

---
### Option A: Docker Compose (Recommended)

```bash
git clone https://github.com/Ishita-rastogi06/AgentHive.git
cd AgentHive
docker-compose up --build
```

Open **`http://localhost:8501`**. Starts Streamlit (8501), Redis (6379), and PostgreSQL (host port `5433`) with auto-seeded demo data.

> **Note**: Inside Docker, the app uses `postgres:5432` (hardcoded in `docker-compose.yml`, overrides `.env`). To connect external tools like pgAdmin, use `postgresql://agenthive:agenthive@localhost:5433/agenthive`.

---

### Option B: Native Local

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows
source .venv/bin/activate           # Linux/macOS

pip install -r requirements.txt
cp .env.example .env                # DB_URL should be localhost:5433

docker-compose up -d postgres redis # start once; runs in background
streamlit run app.py
```

---

## 🎬 3-Minute End-to-End Walkthrough

1. Open **`http://localhost:8501`** (Login / Signup page).
2. Sign up or log in with your credentials (stored securely with `bcrypt` in PostgreSQL/SQLite).
3. Navigate to **Workspace**, enter a complex objective (e.g. *"Query the customers table for active accounts and write a summary report"*).
4. Watch the end-to-end execution:
   - Recalls prior domain facts from ChromaDB.
   - Decomposes task across `data` and `writer` specialists.
   - Triggers pre-execution approval pause on non-read-only SQL `INSERT`.
   - Resumes via `Command(resume=...)` from the **Approval Queue**.
   - Reviewer approves, memory persists to ChromaDB.
5. Navigate to **Trace Explorer** to inspect the full interactive execution trace tree!

---

## 🧪 Running the Full Automated Test Suite

AgentHive features a comprehensive test suite covering unit, integration, memory, checkpointer, approval, trace, and resilience scenarios.

```bash
py -m pytest
```

```text
================= 240 passed, 8 warnings in 114.91s (0:01:54) =================
```
