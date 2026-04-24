from typing import Tuple, Any
from qlearning.core.reward import BaseReward
from qlearning.core.config import EnvConfig

class TransportReward(BaseReward):
    """
    Concrete reward implementation for transport tasks.
    Incentivizes efficiency via step costs and rewards delivery success.
    """
    def __init__(self, config: EnvConfig = EnvConfig()):
        self.config = config

    def calculate_reward(self, 
                         previous_state: Tuple[Any, ...], 
                         action: int, 
                         current_state: Tuple[Any, ...], 
                         is_terminal_state: bool) -> float:
        """
        Calculates the scalar reward signal R_{t+1}.
        
        Returns:
            - success_reward if the agent reached the goal.
            - step_cost otherwise (to penalise long paths).
        """
        if is_terminal_state:
            return self.config.success_reward
        return self.config.step_cost
