"""LangGraph wiring for AgentHive's multi-agent workflow."""

from langgraph.graph import END, START, StateGraph

from src.agents.memory_recall import MemoryRecallAgent
from src.agents.memory_save import MemorySaveAgent
from src.agents.reviewer import Reviewer
from src.agents.specialist_coder import CodingSpecialist
from src.agents.specialist_research import ResearchSpecialist
from src.agents.supervisor import Supervisor
from src.llm.ollama_client import OllamaClient
from src.orchestration.state import AgentHiveState


def build_graph():
    """Build and compile the AgentHive task workflow."""
    llm = OllamaClient()

    workflow = StateGraph(AgentHiveState)

    workflow.add_node("memory_recall", MemoryRecallAgent().run)
    workflow.add_node("supervisor", Supervisor(llm).run)
    workflow.add_node("research", ResearchSpecialist(llm).run)
    workflow.add_node("coder", CodingSpecialist(llm).run)
    workflow.add_node("reviewer", Reviewer().run)
    workflow.add_node("memory_save", MemorySaveAgent().run)

    workflow.add_edge(START, "memory_recall")
    workflow.add_edge("memory_recall", "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state.get("selected_agent", "research"),
        {
            "research": "research",
            "coder": "coder",
        },
    )

    workflow.add_edge("research", "reviewer")
    workflow.add_edge("coder", "reviewer")
    workflow.add_edge("reviewer", "memory_save")
    workflow.add_edge("memory_save", END)

    return workflow.compile()