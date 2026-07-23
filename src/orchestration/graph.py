"""LangGraph wiring for AgentHive's first supervisor workflow."""
from langgraph.graph import END, START, StateGraph
from src.agents.supervisor import Supervisor
from src.agents.specialist_research import ResearchSpecialist
from src.agents.specialist_coder import CodingSpecialist
from src.agents.reviewer import Reviewer
from src.llm.ollama_client import OllamaClient
from src.orchestration.state import AgentHiveState

def build_graph():
    llm = OllamaClient()
    supervisor = Supervisor(llm)
    researcher = ResearchSpecialist(llm)
    coder = CodingSpecialist(llm)
    reviewer = Reviewer()

    workflow = StateGraph(AgentHiveState)
    workflow.add_node("supervisor", supervisor.run)
    workflow.add_node("research", researcher.run)
    workflow.add_node("coder", coder.run)
    workflow.add_node("reviewer", reviewer.run)
    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges("supervisor", lambda state: state["selected_agent"], {"research": "research", "coder": "coder"})
    workflow.add_edge("research", "reviewer")
    workflow.add_edge("coder", "reviewer")
    workflow.add_edge("reviewer", END)
    return workflow.compile()
