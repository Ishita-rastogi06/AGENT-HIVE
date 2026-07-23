from src.agents.base import BaseAgent
from src.llm.ollama_client import OllamaClient
from src.orchestration.state import AgentHiveState

class CodingSpecialist(BaseAgent):
    name = "coder"
    def __init__(self, llm: OllamaClient) -> None:
        self.llm = llm

    def run(self, state: AgentHiveState) -> dict:
        response = self.llm.ask(
            state["task"],
            "You are AgentHive's coding specialist. Provide correct, minimal code and explain how to use it. Do not claim to have run code you did not run.",
        ) or "Ollama is not reachable yet. Start Ollama, pull the model configured in .env, and try again."
        return {"specialist_output": response, "trace": [*state.get("trace", []), "Coding specialist completed"]}
