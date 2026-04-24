import logging
from typing import List

class MetricLogger:
    """Handles formatted logging of RL metrics."""
    def __init__(self, name: str = "qlearning"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.episode_rewards: List[float] = []

    def log_episode(self, episode: int, total_reward: float, initial_epsilon: float):
        self.episode_rewards.append(total_reward)
        if episode % 1000 == 0:
            self.logger.info(f"Episode {episode} | Reward: {total_reward:.2f} | Epsilon: {initial_epsilon:.4f}")
