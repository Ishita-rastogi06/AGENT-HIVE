# AgentHive

A local, Streamlit-based multi-agent workspace powered by Ollama and LangGraph.

## Stage 1: run the supervisor skeleton

1. Install [Ollama](https://ollama.com/) and download a model, for example:
   ```bash
   ollama pull llama3.2:3b
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # macOS/Linux
   source .venv/bin/activate
   # Windows PowerShell
   .venv\\Scripts\\Activate.ps1
   ```
3. Install dependencies and configure the model:
   ```bash
   pip install -r requirements.txt
   cp .env.example .env
   ```
4. Start the app from this project root:
   ```bash
   streamlit run app.py
   ```

The first stage routes requests to a research or coding specialist, then sends the response through a reviewer. If Ollama is unavailable, it displays a clear local fallback so the interface remains testable.

## Next stages
- Add Chroma-backed long-term memory
- Add safe workspace file tools and sandbox execution
- Add richer planning, citations, and agent traces
