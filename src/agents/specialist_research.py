from src.agents.base import BaseAgent
from src.llm.ollama_client import OllamaClient
from src.orchestration.state import AgentHiveState

class ResearchSpecialist(BaseAgent):
    name = "research"
    def __init__(self, llm: OllamaClient) -> None:
        self.llm = llm

    def run(self, state: AgentHiveState) -> dict:
        response = self.llm.ask(
            state["task"],
            "You are AgentHive's research specialist. Give a concise, factual answer. State uncertainty plainly; do not invent sources.",
        ) or "Ollama is not reachable yet. Start Ollama, pull the model configured in .env, and try again."
        return {"specialist_output": response, "trace": [*state.get("trace", []), "Research specialist completed"]}
