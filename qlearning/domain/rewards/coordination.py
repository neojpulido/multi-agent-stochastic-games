class CoordinationReward(BaseReward):
    """Assignment 2: Handles Collisions and Stochastic Lake Penalties."""

    def __init__(self, config, phase: int):
        self.cfg = config
        self.phase = phase  # 1 for Sarah (Safe), 2 for Robert (Efficient)

    def get_rewards(self, prev_obs, joint_action, curr_obs):
        rewards = {}
        positions = curr_obs["agent_positions"]
        lake_flooded = curr_obs["lake_flooded"]

        # 1. Global Collision Logic (Robert's primary concern)
        pos_list = list(positions.values())
        has_collision = len(pos_list) != len(set(pos_list))

        for aid, pos in positions.items():
            r = self.cfg.step_cost

            # 2. Hazard Penalty (Sarah's Phase 1 logic)
            # Only apply if the phase is 1 AND the lake is flooded
            if self.phase == 1 and lake_flooded and self._is_in_lake(pos):
                r += self.cfg.hazard_penalty

            # 3. Collision Penalty (Common to both)
            if has_collision:
                r += self.cfg.collision_penalty

            # 4. Success Reward
            if pos == self.cfg.goal_pos and curr_obs["has_payload"][aid]:
                r += self.cfg.success_reward

            rewards[aid] = r
        return rewards