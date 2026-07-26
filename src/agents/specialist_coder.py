"""Coding specialist for AgentHive."""

from src.agents.base import BaseAgent
from src.llm.ollama_client import OllamaClient
from src.orchestration.state import AgentHiveState


class CodingSpecialist(BaseAgent):
    """Handle coding tasks using relevant long-term memory."""

    name = "coder"

    def __init__(self, llm: OllamaClient) -> None:
        self.llm = llm

    def run(self, state: AgentHiveState) -> dict:
        """Write code for the task using recalled memory only when relevant."""
        prompt = state["task"]

        if state.get("memory_context"):
            prompt = f"{prompt}\n\n{state['memory_context']}"

        response = self.llm.ask(
            prompt,
            (
                "You are AgentHive's coding specialist. Provide correct, "
                "minimal code and explain how to use it. Do not claim to "
                "have run code you did not run. Use recalled memory only "
                "when it helps answer the current task."
            ),
        ) or (
            "Ollama is not reachable yet. Start Ollama, pull the model "
            "configured in .env, and try again."
        )

        return {
            "specialist_output": response,
            "trace": [
                *state.get("trace", []),
                "Coding specialist completed",
            ],
        }