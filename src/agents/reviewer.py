from src.agents.base import BaseAgent
from src.orchestration.state import AgentHiveState

class Reviewer(BaseAgent):
    name = "reviewer"
    def run(self, state: AgentHiveState) -> dict:
        # A deterministic reviewer makes the initial workflow reliable; model-based critique comes later.
        answer = state["specialist_output"].strip()
        return {"final_answer": answer, "trace": [*state.get("trace", []), "Reviewer completed"]}
