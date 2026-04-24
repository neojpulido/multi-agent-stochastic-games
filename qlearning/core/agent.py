from abc import ABC, abstractmethod
from typing import Tuple, Any

class BaseAgent(ABC):
    """Abstract Base Class supporting both Single and Multi-Agent RL."""

    def __init__(self, agent_id: str, config: Any):
        self.agent_id = agent_id
        self.config = config

    @abstractmethod
    def choose_action(self, observation: Any) -> int:
        """Selects action based on policy. Observation can be local or joint."""
        pass

    @abstractmethod
    def update_learning(self,
               state: Any,
               action: int,
               reward: float,
               next_state: Any,
               terminal: bool) -> None:
        """Update the Q-values based on the transition."""
        pass
