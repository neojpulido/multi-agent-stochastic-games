from typing import Dict, List, Any
import numpy as np
from src.core.config import ExperimentConfig
from src.core.state import StateHandler

class SimulationRunner:
    """
    Orchestrates the Multi-Agent training loop.
    """

    def __init__(self, config: ExperimentConfig, env: Any, agents: Dict[str, Any]):
        self.config = config
        self.env = env
        self.agents = agents

    def run_experiment(self):
        """
        Executes the training loop over the specified episode budget.
        """
        print(f"--- Starting Experiment: {self.config.experiment_name} ---")

        for episode in range(self.config.training_episode_budget):
            obs = self.env.reset()
            
            episode_dones = {aid: False for aid in self.agents.keys()}
            total_rewards = {aid: 0.0 for aid in self.agents.keys()}
            step_count = 0
            collisions = 0

            while not all(episode_dones.values()):
                # 1. Action Selection (Decentralized)
                joint_action = {}
                for aid, agent in self.agents.items():
                    if not episode_dones[aid]:
                        state_key = StateHandler.get_agent_state(aid, obs)
                        joint_action[aid] = agent.choose_action(state_key)
                    else:
                        # If agent is done, it could technically WAIT or stay still
                        joint_action[aid] = 4 # Directions.WAIT

                # 2. Environment Step
                next_obs, rewards, dones, truncated = self.env.step(joint_action)

                # 3. Learning (Decentralized)
                for aid, agent in self.agents.items():
                    if not episode_dones[aid]:
                        state_key = StateHandler.get_agent_state(aid, obs)
                        next_state_key = StateHandler.get_agent_state(aid, next_obs)
                        
                        agent.update_learning(
                            state_key,
                            joint_action[aid],
                            rewards[aid],
                            next_state_key,
                            dones[aid]
                        )
                        total_rewards[aid] += rewards[aid]
                        
                        # Collision detection for metrics
                        if rewards[aid] <= self.config.env.collision_penalty and self.config.env.collision_penalty < 0:
                            collisions += 1

                obs = next_obs
                episode_dones = dones
                step_count += 1

                if step_count > 2000: # Safety break
                    break

            # 4. Epsilon Decay (Per Episode)
            for agent in self.agents.values():
                agent.decay_epsilon()

            # Logging
            if episode % 500 == 0:
                avg_reward = np.mean(list(total_rewards.values()))
                print(f"Episode {episode:5d} | Avg Reward: {avg_reward:7.2f} | Steps: {step_count:4d} | Collisions: {collisions:3d}")

        print("--- Training Complete ---")
