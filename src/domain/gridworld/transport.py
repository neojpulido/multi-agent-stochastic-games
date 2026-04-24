from typing import Dict, Tuple, Any, Optional
import numpy as np
from src.core.environment import BaseEnvironment
from src.core.config import EnvConfig
from src.core.actions import Actions

class GridWorldTransport(BaseEnvironment):
    """
    Assignment 1 Baseline: Single-agent pickup and delivery task.
    Refactored to be compatible with the Stage 2 Orchestrator.
    """
    def __init__(self, config: EnvConfig):
        self.config = config
        self.grid_size = (self.config.grid_rows, self.config.grid_cols)
        
        # Goal B is at the bottom-right corner
        self.goal_coordinates = (self.config.grid_rows - 1, self.config.grid_cols - 1)
        self.reset()

    def reset(self) -> Dict[str, Any]:
        """
        Resets the world for a new Assignment 1 episode.
        """
        # Random starting coordinates
        self.agent_pos = (np.random.randint(self.config.grid_rows), np.random.randint(self.config.grid_cols))
        
        while True:
            self.pickup_pos = (np.random.randint(self.config.grid_rows), np.random.randint(self.config.grid_cols))
            if self.pickup_pos != self.goal_coordinates:
                break
                
        self.has_payload = False
        self.done = False
        
        return self._get_obs()

    def _get_obs(self) -> Dict[str, Any]:
        """
        Returns observation in the format expected by the Multi-Agent runner.
        """
        return {
            "positions": {"Agent_1": self.agent_pos},
            "has_sample": {"Agent_1": self.has_payload},
            "lake_flooded": False # Not applicable for A1, but required for schema
        }

    def step(self, joint_action: Dict[str, int]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], bool]:
        """
        Executes a step for the single agent (Agent_1).
        """
        action_idx = joint_action.get("Agent_1", 4) # Default to WAIT
        reward = 0.0
        
        if not self.done:
            # 1. Movement
            potential_pos = Actions.apply_action(self.agent_pos, action_idx)
            if Actions.is_valid_move(potential_pos, self.grid_size):
                self.agent_pos = potential_pos
            
            # 2. Costs
            reward += self.config.step_cost
            
            # 3. Pickup Logic
            if not self.has_payload and self.agent_pos == self.pickup_pos:
                self.has_payload = True
                reward += 10.0 # Standard A1 pickup reward
                
            # 4. Delivery Logic
            if self.has_payload and self.agent_pos == self.goal_coordinates:
                self.done = True
                reward += self.config.success_reward

        obs = self._get_obs()
        return obs, {"Agent_1": reward}, {"Agent_1": self.done}, self.done
