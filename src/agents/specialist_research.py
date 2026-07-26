"""Research specialist for AgentHive."""

from src.agents.base import BaseAgent
from src.llm.ollama_client import OllamaClient
from src.orchestration.state import AgentHiveState


class ResearchSpecialist(BaseAgent):
    """Answer research-oriented tasks using relevant long-term memory."""

    name = "research"

    def __init__(self, llm: OllamaClient) -> None:
        self.llm = llm

    def run(self, state: AgentHiveState) -> dict:
        """Research the task using recalled memory only when relevant."""
        prompt = state["task"]

        if state.get("memory_context"):
            prompt = f"{prompt}\n\n{state['memory_context']}"

        response = self.llm.ask(
            prompt,
            (
                "You are AgentHive's research specialist. Give a concise, "
                "factual answer. State uncertainty plainly; do not invent "
                "sources. Use recalled memory only if it helps answer the "
                "current task."
            ),
        ) or (
            "Ollama is not reachable yet. Start Ollama, pull the model "
            "configured in .env, and try again."
        )

        return {
            "specialist_output": response,
            "trace": [
                *state.get("trace", []),
                "Research specialist completed",
            ],
        }