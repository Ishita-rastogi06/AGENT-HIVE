import streamlit as st
from src.config import settings
from src.llm.ollama_client import OllamaClient
from src.orchestration.graph import build_graph

def render_app() -> None:
    settings.ensure_runtime_directories()
    st.set_page_config(page_title="AgentHive", page_icon="🐝", layout="wide")
    st.title("🐝 AgentHive")
    st.caption("A local multi-agent workspace — Stage 1: supervisor routing")

    available = OllamaClient().is_available()
    with st.sidebar:
        st.subheader("Local model")
        st.write(f"**Model:** `{settings.ollama_model}`")
        st.write(f"**Endpoint:** `{settings.ollama_host}`")
        st.success("Ollama connected") if available else st.warning("Ollama unavailable — fallback responses will be shown.")
        st.divider()
        st.caption("Next: persistent memory, workspace tools, and safe code execution.")

    task = st.text_area("What should AgentHive do?", placeholder="Research a topic or ask for help with code.", height=150)
    if st.button("Run AgentHive", type="primary", disabled=not task.strip()):
        with st.spinner("Routing work through the hive..."):
            result = build_graph().invoke({"task": task.strip(), "trace": []})
        st.subheader("Answer")
        st.markdown(result["final_answer"])
        with st.expander("Workflow trace"):
            st.write(" → ".join(result["trace"]))
