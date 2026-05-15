from typing import Dict, List, Any, Optional
import numpy as np
from src.core.config import ExperimentConfig
from src.core.state import StateHandler
from src.orchestration.jupyter_utils import JupyterEvaluator

class SimulationRunner:
    """
    Orchestrates the Multi-Agent training and evaluation loops.
    Supports both Task 1 (Phase 1) and Task 2 (Phase 2) simulations.
    """

    def __init__(self, config: ExperimentConfig, env: Any, agents: Dict[str, Any]):
        self.config = config
        self.env = env
        self.agents = agents
        self.history = {
            "episodes": [],
            "avg_return": [],
            "avg_length": [],
            "collision_rate": [],
            "hazard_rate": []
        }

    def run_experiment(self, eval_interval: int = 500) -> Dict[str, List[float]]:
        """
        Executes the training loop with periodic greedy evaluations.
        """
        print(f"--- Starting Experiment: {self.config.experiment_name} ---")

        for episode in range(self.config.training_episode_budget):
            obs = self.env.reset()
            episode_dones = {aid: False for aid in self.agents.keys()}
            step_count = 0

            # 1. Training Loop
            while not all(episode_dones.values()) and step_count < 1000:
                joint_action = {}
                for aid, agent in self.agents.items():
                    if not episode_dones[aid]:
                        state_key = StateHandler.get_agent_state(aid, obs)
                        joint_action[aid] = agent.choose_action(state_key)
                    else:
                        joint_action[aid] = 4 # Directions.WAIT

                next_obs, rewards, dones, _ = self.env.step(joint_action)

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

                obs = next_obs
                episode_dones = dones
                step_count += 1

            # 2. Epsilon Decay
            for agent in self.agents.values():
                agent.decay_epsilon()

            # 3. Periodic Greedy Evaluation
            if episode % eval_interval == 0:
                metrics = JupyterEvaluator.evaluate_greedy(self.env, self.agents, episodes=20)
                
                self.history["episodes"].append(episode)
                for k, v in metrics.items():
                    self.history[k].append(v)
                
                print(f"Episode {episode:5d} | Return: {metrics['avg_return']:7.2f} | Collisions: {metrics['collision_rate']:5.2f} | Hazards: {metrics['hazard_rate']:5.2f}")

        print("--- Training Complete ---")
        return self.history
