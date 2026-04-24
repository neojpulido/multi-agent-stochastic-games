from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any


class BaseEnvironment(ABC):
    """Abstract Environment for MDPs (Stage 1) and Stochastic Games (Stage 2)."""

    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Resets world. Returns Dict mapping AgentID -> Observation."""
        pass

    @abstractmethod
    def step(self, joint_action: Dict[str, int]) -> Tuple[Dict[str, Any], Dict[str, float], bool]:
        """
        Executes actions for all agents.
        Returns: (Observations, Rewards, Done_Flag)
        """
        pass