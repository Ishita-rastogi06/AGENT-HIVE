from abc import ABC, abstractmethod
from src.orchestration.state import AgentHiveState

class BaseAgent(ABC):
    name: str

    @abstractmethod
    def run(self, state: AgentHiveState) -> dict:
        """Return only state fields that this agent changes."""
