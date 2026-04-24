import random
from typing import Dict, Tuple, Any, List
from src.core.environment import BaseEnvironment
from src.core.config import EnvConfig
from src.core.actions import Actions, Directions

class StochasticMultiAgentEnv(BaseEnvironment):
    """
    Stage 2 GridWorld: Multi-agent, simultaneous steps, stochastic lake.
    Refactored for Industry/Academic standards using Actions class.
    """

    def __init__(self, config: EnvConfig):
        self.config = config
        self.grid_size = (config.grid_rows, config.grid_cols)
        
        # Locations (y, x)
        self.locs = {
            "X": (2, 0),
            "Y": (0, 2),
            "U": (2, 4),
            "V": (4, 2),
            "Lake": (2, 2)
        }
        
        self.agent_types = {
            "Agent_A": "Type_A", # X -> U -> X
            "Agent_B": "Type_B"  # Y -> V -> Y
        }
        
        self.reset()

    def reset(self) -> Dict[str, Any]:
        self.positions = {
            "Agent_A": self.locs["X"],
            "Agent_B": self.locs["Y"]
        }
        self.has_sample = {
            "Agent_A": False,
            "Agent_B": False
        }
        self.lake_flooded = False # Initial state
        self.done = {
            "Agent_A": False,
            "Agent_B": False
        }
        return self._get_obs()

    def _get_obs(self) -> Dict[str, Any]:
        return {
            "positions": self.positions.copy(),
            "has_sample": self.has_sample.copy(),
            "lake_flooded": self.lake_flooded
        }

    def step(self, joint_action: Dict[str, int]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], bool]:
        rewards = {"Agent_A": 0.0, "Agent_B": 0.0}
        
        # 1. Stochastic Lake Transition (happens before movement resolution)
        if random.random() < self.config.p_flood:
            self.lake_flooded = not self.lake_flooded

        # 2. Movement Resolution (Simultaneous)
        prev_positions = self.positions.copy()
        new_positions = {}
        
        for agent_id, action in joint_action.items():
            if self.done[agent_id]:
                new_positions[agent_id] = prev_positions[agent_id]
                continue
                
            # Use Actions class for coordinate math
            potential_pos = Actions.apply_action(prev_positions[agent_id], action)
            
            if Actions.is_valid_move(potential_pos, self.grid_size):
                new_positions[agent_id] = potential_pos
            else:
                new_positions[agent_id] = prev_positions[agent_id]
            
            # Step/Wait Costs
            if action == Directions.WAIT:
                rewards[agent_id] += self.config.step_cost * 0.6 # -3 vs -5
            else:
                rewards[agent_id] += self.config.step_cost # -5

        self.positions = new_positions

        # 3. Collision & Water Hazard Detection
        # Collision: different types in the lake at the same time
        if self.positions["Agent_A"] == self.locs["Lake"] and self.positions["Agent_B"] == self.locs["Lake"]:
            rewards["Agent_A"] += self.config.collision_penalty
            rewards["Agent_B"] += self.config.collision_penalty

        # Water Hazard (Agent A only, only in Phase 1)
        if self.lake_flooded and self.positions["Agent_A"] == self.locs["Lake"]:
            rewards["Agent_A"] += self.config.hazard_penalty

        # 4. Task Progress (Pickup / Delivery)
        for agent_id in ["Agent_A", "Agent_B"]:
            if self.done[agent_id]: continue
            
            pos = self.positions[agent_id]
            atype = self.agent_types[agent_id]
            
            if atype == "Type_A":
                # Pickup at U
                if not self.has_sample[agent_id] and pos == self.locs["U"]:
                    self.has_sample[agent_id] = True
                    rewards[agent_id] += 10.0
                # Deliver at X
                elif self.has_sample[agent_id] and pos == self.locs["X"]:
                    self.done[agent_id] = True
                    rewards[agent_id] += self.config.success_reward
            else: # Type B
                # Pickup at V
                if not self.has_sample[agent_id] and pos == self.locs["V"]:
                    self.has_sample[agent_id] = True
                    rewards[agent_id] += 10.0
                # Deliver at Y
                elif self.has_sample[agent_id] and pos == self.locs["Y"]:
                    self.done[agent_id] = True
                    rewards[agent_id] += self.config.success_reward

        all_done = all(self.done.values())
        return self._get_obs(), rewards, self.done, all_done
