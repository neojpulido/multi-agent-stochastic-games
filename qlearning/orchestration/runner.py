from typing import Dict, List, Any
from qlearning.core.config import ExperimentConfig
from qlearning.core.state import StateHandler


class SimulationRunner:
    """
    The Orchestrator: Manages the lifecycle of an experiment.
    Handles the Training Loop, Epsilon Decay, and Multi-Agent Step Resolution.
    """

    def __init__(self, config: ExperimentConfig, env: Any, agents: Dict[str, Any]):
        self.config = config
        self.env = env
        self.agents = agents
        self.history = {"rewards": [], "steps": [], "epsilon": []}

    def run_experiment(self):
        """Main training loop covering the episode budget."""
        print(f"Starting Experiment: {self.config.experiment_name}")

        for episode in range(self.config.training_episode_budget):
            # 1. Reset Env (Returns initial joint observation)
            agent_ids = list(self.agents.keys())
            obs = self.env.reset(agent_ids=agent_ids)

            done = False
            total_episode_reward = 0
            step_count = 0

            while not done:
                # 2. COLLECT: Get actions from all agents simultaneously
                joint_action = {}
                for aid, agent in self.agents.items():
                    # Flatten observation for the specific agent's perspective
                    state_key = StateHandler.to_tabular_key(aid, obs)
                    joint_action[aid] = agent.select_action(state_key)

                # 3. EXECUTE: Pass joint action to the Stochastic Environment
                next_obs, rewards, done = self.env.step(joint_action)

                # 4. LEARN: Each agent updates its own Q-table (Decentralized)
                for aid, agent in self.agents.items():
                    state_key = StateHandler.to_tabular_key(aid, obs)
                    next_state_key = StateHandler.to_tabular_key(aid, next_obs)

                    agent.update(
                        state_key,
                        joint_action[aid],
                        rewards[aid],
                        next_state_key,
                        done
                    )
                    total_episode_reward += rewards[aid]

                obs = next_obs
                step_count += 1

                # Safety break for infinite loops
                if step_count > 1000: break

            # 5. DECAY: Update exploration rates
            for agent in self.agents.values():
                agent.decay_epsilon()

            # Logging progress
            if episode % 500 == 0:
                avg_reward = total_episode_reward / len(self.agents)
                print(f"Episode {episode} | Avg Reward: {avg_reward:.2f} | Steps: {step_count}")

        print("Experiment Training Complete.")