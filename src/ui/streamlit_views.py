import html
import json
from datetime import datetime
from pathlib import Path

import streamlit as st

from src.config import settings
from src.llm.ollama_client import OllamaClient
from src.orchestration.graph import build_graph


HISTORY_FILE = Path("data/task_history.json")


def apply_custom_styles() -> None:
    """Apply AgentHive dashboard styling."""
    st.markdown(
        """
        <style>
            .block-container {
                max-width: 1180px;
                padding-top: 2.2rem;
                padding-bottom: 3rem;
            }

            h1 {
                font-size: 3rem !important;
                font-weight: 800 !important;
                margin-bottom: 0.2rem !important;
            }

            h2 {
                font-size: 1.7rem !important;
                margin-top: 1.5rem !important;
            }

            p, li, .stMarkdown, label {
                font-size: 1.1rem !important;
                line-height: 1.65 !important;
            }

            textarea {
                font-size: 1.1rem !important;
                line-height: 1.6 !important;
            }

            .stButton button {
                font-size: 1.1rem !important;
                font-weight: 700 !important;
                padding: 0.65rem 1.4rem !important;
                border-radius: 10px !important;
            }

            .agent-card {
                border: 1px solid #d9dce7;
                border-radius: 14px;
                padding: 18px;
                min-height: 128px;
                background: #ffffff;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
            }

            .agent-title {
    font-size: 1.15rem;
    font-weight: 750;
    margin-bottom: 8px;
    color: #1e293b;
}

            .agent-description {
                font-size: 0.98rem;
                color: #5b6275;
                line-height: 1.45;
            }

            .status-ready {
                color: #737b8c;
                font-weight: 700;
            }

            code {
                font-size: 1rem !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_task_history() -> list[dict]:
    """Load locally saved AgentHive task history."""
    if not HISTORY_FILE.exists():
        return []

    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as history_file:
            saved_history = json.load(history_file)

        return saved_history if isinstance(saved_history, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_task_history(history: list[dict]) -> None:
    """Save AgentHive task history locally."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    with HISTORY_FILE.open("w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2, ensure_ascii=False)


def render_agent_card(
    title: str,
    icon: str,
    description: str,
    status: str,
    status_class: str,
) -> None:
    """Display one dashboard card for an agent."""
    st.markdown(
        f"""
        <div class="agent-card">
            <div class="agent-title">{icon} {title}</div>
            <div class="agent-description">{description}</div>
            <br>
            <div class="{status_class}">{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_history_page() -> None:
    """Render the saved task-history page."""
    history = load_task_history()

    st.title("📜 Task History")
    st.caption("Previously completed AgentHive tasks saved on this computer.")

    if not history:
        st.info("No saved tasks yet. Run a task from the AgentHive Workspace.")
        return

    top_col, clear_col = st.columns([4, 1])

    with top_col:
        st.write(f"**{len(history)} saved task(s)**")

    with clear_col:
        if st.button("Clear All History", type="secondary"):
            save_task_history([])
            st.rerun()

    st.download_button(
        label="Download History",
        data=json.dumps(history, indent=2, ensure_ascii=False),
        file_name="agenthive_task_history.json",
        mime="application/json",
    )

    st.divider()

    for item_number, item in enumerate(reversed(history), start=1):
        selected_agent = item.get("selected_agent", "research")
        agent_label = (
            "💻 Coding Specialist"
            if selected_agent == "coder"
            else "🔎 Research Specialist"
        )

        timestamp = item.get("timestamp", "Unknown time")
        task = item.get("task", "No task saved")
        trace = item.get("trace", [])
        final_answer = item.get("final_answer", "No answer saved")

        with st.expander(
            f"#{item_number} · {agent_label} · {timestamp}",
            expanded=False,
        ):
            st.markdown("### Task")
            st.write(task)

            st.markdown("### Workflow")
            if trace:
                st.write(" → ".join(trace))
            else:
                st.caption("No workflow trace saved.")

            st.markdown("### Final Answer")
            st.markdown(final_answer)


def go_to_search() -> None:
    """Open the Search page from the Dashboard button."""
    st.session_state.page_navigation = "Search"


def render_dashboard_page() -> None:
    """Render the AgentHive dashboard only."""
    st.title("🐝 AgentHive Dashboard")
    st.caption("A local multi-agent workspace for research and coding tasks")

    st.subheader("Agent Team")

    supervisor_col, specialist_col, reviewer_col = st.columns(3)

    with supervisor_col:
        render_agent_card(
            title="Supervisor",
            icon="🧭",
            description="Reads the task and chooses the right specialist.",
            status="● Ready",
            status_class="status-ready",
        )

    with specialist_col:
        render_agent_card(
            title="Research / Coding Specialist",
            icon="🧠",
            description="Researches a topic or creates coding guidance.",
            status="● Waiting for task",
            status_class="status-ready",
        )

    with reviewer_col:
        render_agent_card(
            title="Reviewer",
            icon="✅",
            description="Checks the specialist output before final delivery.",
            status="● Waiting for task",
            status_class="status-ready",
        )

    st.divider()

    st.subheader("Ready to work with the hive?")
    st.write(
        "Ask a research question, request Python code, or explore a technical topic. "
        "AgentHive will select the right specialist and review the response."
    )

    st.button(
        "🔎 Start a New Search",
        type="primary",
        on_click=go_to_search,
    )


def render_search_page() -> None:
    """Render the page where the user gives AgentHive a task."""
    st.title("🔎 Give AgentHive a Task")
    st.caption("Describe what you want to research, create, or understand.")

    task = st.text_area(
        "What should AgentHive do?",
        placeholder=(
            "Research example: Explain the difference between REST APIs and GraphQL.\n\n"
            "Coding example: Write a Python function to check if a string is a palindrome."
        ),
        height=180,
    )

    if st.button("Run AgentHive", type="primary", disabled=not task.strip()):
        with st.spinner("AgentHive is processing your task..."):
            result = build_graph().invoke(
                {
                    "task": task.strip(),
                    "trace": [],
                }
            )

        selected_agent = result.get("selected_agent", "research")
        history = load_task_history()

        history.append(
            {
                "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
                "task": task.strip(),
                "selected_agent": selected_agent,
                "trace": result.get("trace", []),
                "final_answer": result.get(
                    "final_answer",
                    "No answer was returned.",
                ),
            }
        )

        save_task_history(history)

        if selected_agent == "coder":
            selected_agent_name = "Coding Specialist"
            selected_agent_icon = "💻"
        else:
            selected_agent_name = "Research Specialist"
            selected_agent_icon = "🔎"

        st.divider()
        st.subheader("Task Processing Summary")

        summary_col_1, summary_col_2, summary_col_3 = st.columns(3)

        with summary_col_1:
            st.metric("Selected Agent", selected_agent_name)

        with summary_col_2:
            st.metric("Task Type", selected_agent.title())

        with summary_col_3:
            st.metric("Review Status", "Completed")

        st.success(
            f"{selected_agent_icon} The Supervisor selected the "
            f"**{selected_agent_name}** for this task."
        )

        st.subheader("Final Answer")
        st.markdown(result.get("final_answer", "No answer was returned."))

        st.subheader("Workflow Trace")
        trace = result.get("trace", [])

        if trace:
            st.caption("Your task passed through these completed steps:")

            for step_number, step in enumerate(trace, start=1):
                safe_step = html.escape(step)

                st.markdown(
                    f"""
                    <div style="
                        display: flex;
                        align-items: center;
                        gap: 14px;
                        margin: 10px 0;
                        padding: 14px 18px;
                        border: 1px solid #c7d7fe;
                        border-left: 5px solid #2563eb;
                        border-radius: 10px;
                        background-color: #eff6ff;
                        color: #172554;
                        font-size: 17px;
                        font-weight: 650;
                    ">
                        <span style="
                            display: inline-flex;
                            align-items: center;
                            justify-content: center;
                            width: 30px;
                            height: 30px;
                            min-width: 30px;
                            border-radius: 50%;
                            background-color: #2563eb;
                            color: #ffffff;
                            font-size: 15px;
                            font-weight: 800;
                        ">
                            {step_number}
                        </span>
                        <span style="color: #172554;">{safe_step}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No workflow trace is available.")

def render_app() -> None:
    """Configure and render the AgentHive application."""
    settings.ensure_runtime_directories()

    st.set_page_config(
        page_title="AgentHive",
        page_icon="🐝",
        layout="wide",
    )

    apply_custom_styles()

    available = OllamaClient().is_available()

    with st.sidebar:
        st.title("🐝 AgentHive")

        page = st.radio(
    "Navigation",
    options=["Dashboard", "Search", "Task History"],
    key="page_navigation",
    label_visibility="collapsed",
)

        st.divider()
        st.subheader("System Status")
        st.write(f"**Model:** `{settings.ollama_model}`")
        st.write(f"**Endpoint:** `{settings.ollama_host}`")

        if available:
            st.success("Ollama connected")
        else:
            st.warning("Ollama unavailable")

        st.divider()
        st.subheader("How it works")
        st.write("1. Supervisor routes your task")
        st.write("2. Specialist prepares an answer")
        st.write("3. Reviewer returns the final response")

    if page == "Dashboard":
        render_dashboard_page()
    elif page == "Search":
        render_search_page()
    else:
        render_history_page()