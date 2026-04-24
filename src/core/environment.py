from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any

class BaseEnvironment(ABC):
    """
    Abstract Base for Multi-Agent Stochastic Games.
    Supports simultaneous steps and heterogeneous agents.
    """

    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Resets the environment. Returns initial observation dict."""
        pass

    @abstractmethod
    def step(self, joint_action: Dict[str, int]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], bool]:
        """
        Executes a joint action.
        Returns:
            observations: Dict[AgentID, Any]
            rewards: Dict[AgentID, float]
            dones: Dict[AgentID, bool]
            truncated: bool
        """
        pass
