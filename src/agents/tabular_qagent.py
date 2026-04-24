from collections import defaultdict
from typing import Tuple, Any
import numpy as np
from src.core.agent import BaseAgent
from src.core.config import AgentConfig

class TabularQAgent(BaseAgent):
    """
    Agent implementation using the Tabular Q-Learning algorithm.
    Refactored to match Stage 2 Multi-Agent requirements.
    """

    def __init__(self, agent_id: str, config: AgentConfig):
        super().__init__(agent_id, config)
        
        # Q-Table Initialization: Q(s, a)
        # Using a sparse mapping (defaultdict)
        self.q_table = defaultdict(lambda: np.zeros(self.config.action_size))

        # Exploration rate (epsilon)
        self.epsilon = self.config.initial_epsilon

    def choose_action(self, observation: Any) -> int:
        """
        Epsilon-greedy action selection.
        Observation should be the hashable state tuple from StateHandler.
        """
        if np.random.rand() < self.epsilon:
            return int(np.random.randint(self.config.action_size))

        return int(np.argmax(self.q_table[observation]))

    def update_learning(self,
                        state: Any,
                        action: int,
                        reward: float,
                        next_state: Any,
                        terminal: bool) -> None:
        """
        Tabular Q-Learning update rule:
        Q(s, a) ← Q(s, a) + α * [R + γ * max_a' Q(s', a') - Q(s, a)]
        """
        current_q = self.q_table[state][action]
        
        if terminal:
            max_future_q = 0.0
        else:
            max_future_q = np.max(self.q_table[next_state])

        # Bellman update
        td_target = reward + (self.config.discount_factor_gamma * max_future_q)
        self.q_table[state][action] += self.config.learning_rate_alpha * (td_target - current_q)

    def decay_epsilon(self):
        """Standard exponential decay of the exploration rate."""
        self.epsilon = max(
            self.config.minimum_epsilon,
            self.epsilon * self.config.epsilon_decay_rate
        )
