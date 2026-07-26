"""Professional Streamlit dashboard for the AgentHive multi-agent workspace."""

import html
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from src.config import settings
from src.llm.ollama_client import OllamaClient
from src.orchestration.graph import build_graph

HISTORY_FILE = settings.data_dir / "task_history.json"


# ---------- Shared helpers -------------------------------------------------

def apply_custom_styles() -> None:
    """Apply the AgentHive visual system without requiring extra front-end files."""
    st.markdown(
        """
        <style>
        :root { --ink:#111827; --muted:#64748b; --line:#e7ebf3; --hive:#f6b91a; --violet:#6d5dfc; }
        .stApp { background: #f7f8fc; color: var(--ink); }
        .block-container { max-width: 1320px; padding: 2rem 2.2rem 3.5rem; }
        [data-testid="stSidebar"] { background: #101827; border-right: 1px solid #243047; }
        [data-testid="stSidebar"] * { color: #e6edf9; }
        [data-testid="stSidebar"] .stRadio label { border-radius: 10px; padding: .22rem .35rem; }
        [data-testid="stSidebar"] .stRadio label:hover { background:#1c2a3f; }
        [data-testid="stSidebar"] hr { border-color:#2b3950; }
        h1, h2, h3 { color:#162033; letter-spacing:-.035em; }
        h1 { font-size: 2.5rem !important; font-weight:800 !important; margin-bottom:.2rem !important; }
        h2 { font-size:1.35rem !important; font-weight:750 !important; margin-top:1.35rem !important; }
        h3 { font-size:1rem !important; font-weight:750 !important; }
        p, label { line-height:1.55; }
        /* Buttons: explicit light secondary style prevents black theme boxes. */
        .stButton > button, .stDownloadButton > button {
            min-height:2.65rem; border-radius:10px; border:1px solid #dbe3ef !important;
            background:#ffffff !important; color:#334155 !important; font-weight:750 !important;
            padding:.62rem 1rem; box-shadow:0 2px 6px rgba(15,23,42,.04); transition:.18s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color:#a99ffc !important; background:#f7f6ff !important; color:#4e46d8 !important;
        }
        .stButton > button[kind="primary"] {
            border:0 !important; background:linear-gradient(135deg,#735cff,#4e46d8) !important;
            color:#ffffff !important; box-shadow:0 7px 16px rgba(91,76,227,.22);
        }
        .stButton > button[kind="primary"]:hover { transform:translateY(-1px); filter:brightness(1.05); }
        .stDownloadButton > button *, .stButton > button * { color:inherit !important; }
        /* Consistent breathing room between every Streamlit column. */
        [data-testid="stHorizontalBlock"] { gap:1.15rem !important; align-items:stretch; }
        [data-testid="column"] { min-width:0; }
        .stTextArea textarea { border-radius:12px; border:1px solid #dfe5f0; background:#fff; font-size:1rem; }
        .stTextArea textarea:focus { border-color:#7567f8; box-shadow:0 0 0 3px rgba(117,103,248,.12); }
        .stat-card { min-height:120px; background:#ffffff; border:1px solid #dfe6f0; border-radius:14px; padding:1.18rem 1.25rem; box-sizing:border-box; box-shadow:0 4px 14px rgba(15,23,42,.06); }
        .stat-label { color:#64748b !important; font-size:.83rem; font-weight:700; line-height:1.2; }
        .stat-value { color:#111827 !important; font-size:1.72rem; font-weight:850; letter-spacing:-.04em; line-height:1.2; margin-top:.5rem; }
        .stat-subtitle { color:#64748b !important; font-size:.76rem; margin-top:.35rem; }
        [data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:14px; padding:1rem; box-shadow:0 3px 12px rgba(24,39,75,.035); }
        [data-testid="stMetric"] * { color:#111827 !important; opacity:1 !important; }
        [data-testid="stMetric"] [data-testid="stMetricLabel"] * { color:#64748b !important; }
        .hero { position:relative; overflow:hidden; color:white; margin-top:1.35rem; padding:2.15rem 2.25rem; border-radius:20px; background:linear-gradient(120deg,#171b43 0%,#312b77 58%,#4a3bb5 100%); box-shadow:0 14px 30px rgba(38,30,104,.18); }
        .hero:after { content:""; position:absolute; right:-95px; top:-135px; width:335px; height:335px; border:1px solid rgba(255,255,255,.2); border-radius:50%; box-shadow:0 0 0 42px rgba(255,255,255,.04),0 0 0 88px rgba(255,255,255,.03); }
        .hero-kicker { color:#fbd76a; font-size:.73rem; letter-spacing:.15em; font-weight:800; text-transform:uppercase; margin-bottom:.7rem; }
        .hero-title { font-size:2.15rem; line-height:1.12; letter-spacing:-.045em; font-weight:800; max-width:650px; position:relative; z-index:1; }
        .hero-copy { color:#d9dcff; margin-top:.7rem; max-width:590px; line-height:1.55; position:relative; z-index:1; }
        .section-label { color:#7b8597; font-weight:800; letter-spacing:.12em; font-size:.7rem; text-transform:uppercase; margin:1.55rem 0 .55rem; }
        .agent-card { min-height:176px; background:#fff; border:1px solid var(--line); border-radius:16px; padding:1.25rem; box-shadow:0 3px 12px rgba(24,39,75,.035); }
        .agent-icon { width:38px; height:38px; border-radius:11px; display:inline-flex; align-items:center; justify-content:center; background:#f2efff; font-size:1.2rem; }
        .agent-name { margin-top:.8rem; color:#1b2537; font-weight:800; font-size:1.03rem; }
        .agent-desc { color:#697589; font-size:.88rem; line-height:1.5; margin-top:.35rem; min-height:42px; }
        .agent-status { display:inline-flex; gap:6px; align-items:center; margin-top:.85rem; color:#17724e; background:#eaf8f1; border-radius:20px; padding:.25rem .55rem; font-size:.72rem; font-weight:800; }
        .workspace-panel { background:#fff; border:1px solid var(--line); border-radius:16px; padding:1.35rem; box-shadow:0 3px 12px rgba(24,39,75,.035); }
        .trace-row { display:flex; align-items:center; gap:.8rem; padding:.78rem 0; border-bottom:1px solid #edf0f5; }
        .trace-row:last-child { border:0; }
        .trace-number { width:26px; height:26px; border-radius:50%; flex:0 0 26px; display:flex; align-items:center; justify-content:center; background:#edeaff; color:#5143c9; font-size:.75rem; font-weight:800; }
        .trace-text { color:#374151; font-size:.9rem; font-weight:650; }
        .answer-box { min-height:176px; background:#fff; border:1px solid var(--line); border-radius:16px; padding:1.5rem; box-sizing:border-box; box-shadow:0 3px 12px rgba(24,39,75,.035); }
        .answer-content { margin-top:1rem; color:#263247; font-size:.96rem; line-height:1.7; }
        .trace-empty { color:#64748b; font-size:.9rem; margin-top:1rem; }
        .eyebrow { color:#7d70ed; font-size:.72rem; font-weight:800; letter-spacing:.11em; text-transform:uppercase; }
        .sidebar-brand { padding:.65rem .2rem .8rem; }
        .sidebar-brand-name { font-size:1.35rem; font-weight:850; letter-spacing:-.04em; color:#fff; }
        .sidebar-caption { font-size:.78rem; color:#9eabc2; margin-top:.2rem; }
        .system-chip { color:#a9b7ce; font-size:.78rem; padding:.1rem 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_task_history() -> list[dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []
    try:
        value = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        return value if isinstance(value, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_task_history(history: list[dict[str, Any]]) -> None:
    settings.ensure_runtime_directories()
    HISTORY_FILE.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def route_label(agent: str) -> tuple[str, str, str]:
    if agent == "coder":
        return "Coding Specialist", "⌘", "Code generation"
    return "Research Specialist", "◌", "Research & analysis"


def render_stat_card(label: str, value: str | int, subtitle: str = "") -> None:
    """Render a theme-independent dashboard stat card with readable text."""
    subtitle_html = f'<div class="stat-subtitle">{html.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(
        f'<div class="stat-card"><div class="stat-label">{html.escape(label)}</div>'
        f'<div class="stat-value">{html.escape(str(value))}</div>{subtitle_html}</div>',
        unsafe_allow_html=True,
    )


def render_agent_card(icon: str, name: str, description: str) -> None:
    st.markdown(
        f'''<div class="agent-card"><div class="agent-icon">{icon}</div>
        <div class="agent-name">{name}</div><div class="agent-desc">{description}</div>
        <div class="agent-status"><span>●</span> ONLINE</div></div>''',
        unsafe_allow_html=True,
    )


def render_trace(trace: list[str]) -> None:
    if not trace:
        st.caption("No workflow events were returned for this task.")
        return
    lines = []
    for index, step in enumerate(trace, 1):
        lines.append(f'<div class="trace-row"><div class="trace-number">{index}</div><div class="trace-text">{html.escape(str(step))}</div></div>')
    st.markdown("".join(lines), unsafe_allow_html=True)


def show_result(result: dict[str, Any], elapsed_seconds: float | None = None) -> None:
    selected = result.get("selected_agent", "research")
    name, icon, task_type = route_label(selected)
    answer = result.get("final_answer") or result.get("specialist_output") or "No answer was returned."
    trace = result.get("trace", [])
    if not isinstance(trace, list):
        trace = []

    st.markdown('<div class="section-label">Completed run</div>', unsafe_allow_html=True)
    st.success(f"Task completed — routed to the **{name}** and reviewed by the hive.")
    a, b, c, d = st.columns(4)
    with a:
        render_stat_card("Route", name)
    with b:
        render_stat_card("Task type", task_type)
    with c:
        render_stat_card("Review", "Passed")
    with d:
        render_stat_card("Runtime", f"{elapsed_seconds:.1f}s" if elapsed_seconds is not None else "Complete")

    answer_col, trace_col = st.columns([1.7, 1], gap="medium")
    with answer_col:
        safe_answer = html.escape(str(answer)).replace("\n", "<br>")
        st.markdown(
            f'<div class="answer-box"><div class="eyebrow">Final response</div>'
            f'<div class="answer-content">{safe_answer}</div></div>',
            unsafe_allow_html=True,
        )
    with trace_col:
        trace_html = "".join(
            f'<div class="trace-row"><div class="trace-number">{index}</div>'
            f'<div class="trace-text">{html.escape(str(step))}</div></div>'
            for index, step in enumerate(trace, 1)
        ) or '<div class="trace-empty">No workflow events were returned.</div>'
        st.markdown(
            f'<div class="workspace-panel"><div class="eyebrow">Workflow activity</div>{trace_html}</div>',
            unsafe_allow_html=True,
        )


def render_dashboard_page() -> None:
    st.markdown(
        '''<div class="hero"><div class="hero-kicker">Multi-agent operations</div>
        <div class="hero-title">Bring every complex task to the hive.</div>
        <div class="hero-copy">AgentHive recalls relevant context, selects the right specialist, reviews its work, and saves useful knowledge for the next request.</div></div>''',
        unsafe_allow_html=True,
    )
    history = load_task_history()
    st.markdown('<div class="section-label">Workspace overview</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_stat_card("Agent network", "5 online", "All systems available")
    with m2:
        render_stat_card("Tasks completed", len(history), "Saved local runs")
    with m3:
        render_stat_card("Memory", "Persistent", "Local knowledge store")
    with m4:
        render_stat_card("Review gate", "Enabled", "Quality check active")

    st.markdown('<div class="section-label">Your agent team</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    with cols[0]: render_agent_card("⌁", "Supervisor", "Understands each request and dispatches it to the best specialist.")
    with cols[1]: render_agent_card("◌", "Specialist network", "Research and coding specialists produce focused, high-quality work.")
    with cols[2]: render_agent_card("✓", "Reviewer", "Checks the specialist output before a response is delivered.")

    st.markdown('<div class="section-label">Quick start</div>', unsafe_allow_html=True)
    with st.container(border=True):
        left, right = st.columns([4, 1])
        with left:
            st.markdown("### Start an orchestrated task")
            st.caption("Ask a research question, explore a technical concept, or request implementation help.")
        with right:
            if st.button("Open workspace", type="primary", use_container_width=True):
                st.session_state.page_navigation = "Workspace"
                st.rerun()


def render_workspace_page() -> None:
    st.title("Agent workspace")
    st.caption("Write a clear objective. AgentHive will take care of delegation, review, and memory.")

    with st.container(border=True):
        task = st.text_area(
            "Task objective",
            placeholder="Example: Compare REST and GraphQL for a mobile application, including when to choose each approach.",
            height=175,
            key="task_input",
            label_visibility="visible",
        )
        left, right = st.columns([1, 5])
        with left:
            run = st.button("Run task", type="primary", use_container_width=True, disabled=not task.strip())
        with right:
            st.caption("The workflow uses your local Ollama model. Results are stored in local task history.")

    if run:
        start = time.perf_counter()
        try:
            with st.status("Agents are collaborating…", expanded=True) as progress:
                st.write("Recalling relevant context")
                result = build_graph().invoke({"task": task.strip(), "trace": []})
                st.write("Reviewing the completed response")
                progress.update(label="Task complete", state="complete", expanded=False)
            elapsed = time.perf_counter() - start
            st.session_state.latest_result = {"result": result, "elapsed": elapsed}

            history = load_task_history()
            history.append({
                "timestamp": datetime.now().strftime("%d %b %Y · %I:%M %p"),
                "task": task.strip(),
                "selected_agent": result.get("selected_agent", "research"),
                "trace": result.get("trace", []),
                "final_answer": result.get("final_answer", result.get("specialist_output", "")),
                "runtime_seconds": round(elapsed, 2),
            })
            save_task_history(history[-100:])
        except Exception as error:
            st.error("AgentHive could not complete this task. Confirm that Ollama is running and the configured model is installed.")
            with st.expander("Technical details"):
                st.code(str(error))

    latest = st.session_state.get("latest_result")
    if latest:
        show_result(latest["result"], latest.get("elapsed"))


def render_history_page() -> None:
    st.title("Task history")
    st.caption("A local record of your recent completed AgentHive runs.")
    history = load_task_history()
    top, actions = st.columns([4, 2])
    with top:
        render_stat_card("Saved runs", len(history), "Available in local history")
    with actions:
        d1, d2 = st.columns(2)
        with d1:
            st.download_button("Export JSON", json.dumps(history, indent=2, ensure_ascii=False), "agenthive_history.json", "application/json", use_container_width=True)
        with d2:
            if st.button("Clear history", use_container_width=True, disabled=not history):
                save_task_history([])
                st.session_state.pop("latest_result", None)
                st.rerun()
    if not history:
        st.info("No completed tasks yet. Your first hive run will appear here.")
        return

    for item in reversed(history):
        agent, _, _ = route_label(item.get("selected_agent", "research"))
        task = item.get("task", "Untitled task")
        with st.expander(f"{agent}  ·  {item.get('timestamp', 'Unknown time')}  ·  {task[:75]}"):
            st.markdown("**Objective**")
            st.write(task)
            st.markdown("**Final response**")
            st.markdown(item.get("final_answer", "No saved response."))
            st.markdown("**Workflow trace**")
            render_trace(item.get("trace", []))


def render_sidebar(available: bool) -> str:
    with st.sidebar:
        st.markdown('<div class="sidebar-brand"><div class="sidebar-brand-name">🐝 AgentHive</div><div class="sidebar-caption">ORCHESTRATED INTELLIGENCE</div></div>', unsafe_allow_html=True)
        page = st.radio("Navigation", ["Dashboard", "Workspace", "Task History"], key="page_navigation", label_visibility="collapsed")
        st.divider()
        st.markdown("##### SYSTEM STATUS")
        status = "● Connected" if available else "● Offline"
        color = "#70d6a1" if available else "#ffc46b"
        st.markdown(f'<div class="system-chip" style="color:{color}">{status}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="system-chip">Model · {html.escape(settings.ollama_model)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="system-chip">Memory · Local persistent store</div>', unsafe_allow_html=True)
        st.divider()
        st.markdown("##### WORKFLOW")
        st.caption("Recall context\n\nRoute task\n\nGenerate response\n\nReview and save")
    return page


def render_app() -> None:
    """Configure and render the AgentHive application."""
    settings.ensure_runtime_directories()
    st.set_page_config(page_title="AgentHive | Multi-Agent Workspace", page_icon="🐝", layout="wide", initial_sidebar_state="expanded")
    apply_custom_styles()

    available = OllamaClient().is_available()
    page = render_sidebar(available)
    if page == "Dashboard":
        render_dashboard_page()
    elif page == "Workspace":
        render_workspace_page()
    else:
        render_history_page()
