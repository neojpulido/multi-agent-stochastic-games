from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseReward(ABC):
    """Interface for joint-state reward calculation logic."""

    @abstractmethod
    def calculate_rewards(self,
                          prev_state: Dict[str, Any],
                          joint_action: Dict[str, int],
                          curr_state: Dict[str, Any]) -> Dict[str, float]:
        """Computes a reward for each agent in the system."""
        pass