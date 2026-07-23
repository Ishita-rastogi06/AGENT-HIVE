from src.agents.base import BaseAgent
from src.llm.ollama_client import OllamaClient
from src.orchestration.state import AgentHiveState


class Supervisor(BaseAgent):
    name = "supervisor"

    def __init__(self, llm: OllamaClient) -> None:
        self.llm = llm

    def run(self, state: AgentHiveState) -> dict:
        task = state["task"].lower()

        coding_signals = [
            "write code",
            "write a python",
            "write a script",
            "create a function",
            "debug",
            "error",
            "exception",
            "fix this code",
            "implement",
        ]

        research_signals = [
            "explain",
            "what is",
            "what are",
            "why",
            "compare",
            "difference between",
            "advantages",
            "limitations",
            "plan to learn",
            "research",
            "summarize",
        ]

        if any(word in task for word in coding_signals):
            agent = "coder"

        elif any(word in task for word in research_signals):
            agent = "research"

        else:
            decision = self.llm.ask(
                f"Request: {task}\nChoose exactly one word: research or coder.",
                "You are a router. Reply with only research or coder."
            )

            normalized = (decision or "").strip().lower()

            if normalized == "coder":
                agent = "coder"
            else:
                agent = "research"

        return {
            "selected_agent": agent,
            "trace": [
                *state.get("trace", []),
                f"Supervisor → {agent}"
            ],
        }