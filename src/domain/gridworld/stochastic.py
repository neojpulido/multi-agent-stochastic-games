import random
from typing import Dict, Tuple, Any
from src.core.environment import BaseEnvironment
from src.core.config import EnvConfig
from src.core.actions import Actions, Directions


class StochasticMultiAgentEnv(BaseEnvironment):
    """
    Stage 2 GridWorld with correct simultaneous-step physics:
      1. Lake flips stochastically (before movement)
      2. All agents move simultaneously based on previous positions
      3. Task completion evaluated (sets done flags)
      4. Collision checked only for agents still active after step 3
      5. Water hazard checked only for Agent A if still active
    """

    def __init__(self, config: EnvConfig):
        self.config = config
        self.grid_size = (config.grid_rows, config.grid_cols)
        
        self.locs = {
            "X": (2, 0), 
            "Y": (0, 2),
            "U": (2, 4), 
            "V": (4, 2),
            "Lake": (2, 2)
        }
        
        self.agent_types = {
            "Agent_A": "Type_A", 
            "Agent_B": "Type_B"
        }
        
        self.reset()

    def reset(self) -> Dict[str, Any]:
        self.positions = {"Agent_A": self.locs["X"], "Agent_B": self.locs["Y"]}
        self.has_sample = {"Agent_A": False, "Agent_B": False}
        self.lake_flooded = False
        self.done = {"Agent_A": False, "Agent_B": False}
        return self._get_obs()

    def _get_obs(self) -> Dict[str, Any]:
        return {
            "positions": self.positions.copy(),
            "has_sample": self.has_sample.copy(),
            "lake_flooded": self.lake_flooded
        }

    def step(self, joint_action: Dict[str, int]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, bool], bool]:
        rewards = {"Agent_A": 0.0, "Agent_B": 0.0}

        # 1. Stochastic lake transition (before movement)
        if random.random() < self.config.p_flood:
            self.lake_flooded = not self.lake_flooded

        # 2. Simultaneous movement (based on previous positions)
        prev_positions = self.positions.copy()
        new_positions = {}
        for agent_id, action in joint_action.items():
            if self.done[agent_id]:
                new_positions[agent_id] = prev_positions[agent_id]
                rewards[agent_id] = 0.0
                continue
            candidate = Actions.apply_action(prev_positions[agent_id], action)
            if Actions.is_valid_move(candidate, self.grid_size):
                new_positions[agent_id] = candidate
            else:
                new_positions[agent_id] = prev_positions[agent_id]
            rewards[agent_id] = (self.config.wait_cost if action == Directions.WAIT
                                 else self.config.step_cost)
        self.positions = new_positions

        # 3. Task completion (BEFORE collision — done flags must be set first)
        for agent_id in ["Agent_A", "Agent_B"]:
            if self.done[agent_id]:
                continue
            pos = self.positions[agent_id]
            atype = self.agent_types[agent_id]
            if atype == "Type_A":
                if not self.has_sample[agent_id] and pos == self.locs["U"]:
                    self.has_sample[agent_id] = True
                    rewards[agent_id] += self.config.pickup_reward
                elif self.has_sample[agent_id] and pos == self.locs["X"]:
                    self.done[agent_id] = True
                    rewards[agent_id] += self.config.success_reward
            else:
                if not self.has_sample[agent_id] and pos == self.locs["V"]:
                    self.has_sample[agent_id] = True
                    rewards[agent_id] += self.config.pickup_reward
                elif self.has_sample[agent_id] and pos == self.locs["Y"]:
                    self.done[agent_id] = True
                    rewards[agent_id] += self.config.success_reward

        # 4. Collision (only agents still active after task completion)
        if not self.done["Agent_A"] and not self.done["Agent_B"]:
            if self.positions["Agent_A"] == self.positions["Agent_B"]:
                rewards["Agent_A"] += self.config.collision_penalty
                rewards["Agent_B"] += self.config.collision_penalty

        # 5. Water hazard — Agent A only, only if still active
        if (not self.done["Agent_A"]
                and self.lake_flooded
                and self.positions["Agent_A"] == self.locs["Lake"]):
            rewards["Agent_A"] += self.config.hazard_penalty

        all_done = all(self.done.values())
        return self._get_obs(), rewards, self.done.copy(), all_done
