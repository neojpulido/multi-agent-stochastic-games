from collections import defaultdict
from typing import Tuple, Any, Dict
import numpy as np
from src.core.agent import BaseAgent
from src.core.config import AgentConfig

class TabularQAgent(BaseAgent):
    """
    HD-Compliant Tabular Q-Learning Agent.
    Implements Sarah's 'Expected Value' update by using the direct p_flood parameter.
    """

    def __init__(self, agent_id: str, config: AgentConfig, p_flood: float = 0.5):
        super().__init__(agent_id, config)
        self.p_flood = p_flood
        self.q_table = defaultdict(lambda: np.zeros(self.config.action_size))
        self.epsilon = self.config.initial_epsilon

    def choose_action(self, observation: Tuple, greedy: bool = False) -> int:
        if not greedy and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.config.action_size))

        return int(np.argmax(self.q_table[observation]))

    def update_learning(self,
                        state: Tuple,
                        action: int,
                        reward: float,
                        next_obs_state: Tuple,
                        terminal: bool) -> None:
        """
        Sarah's Refined Bellman Update (HD Implementation).
        Calculates expected value over the estimated transition dynamics.
        """
        current_lake = state[3]
        y_next, x_next, payload_next, _ = next_obs_state
        current_q = self.q_table[state][action]
        
        if terminal:
            expected_future_q = 0.0
        else:
            p = self.p_flood
            
            # The two possible future universes for the lake state
            state_stays = (y_next, x_next, payload_next, current_lake)
            state_flips = (y_next, x_next, payload_next, not current_lake)
            
            # Expected Future Value Equation:
            # E[max Q'] = P(stay) * max Q(S'_stay) + P(flip) * max Q(S'_flip)
            expected_future_q = ((1 - p) * np.max(self.q_table[state_stays])
                                 + p      * np.max(self.q_table[state_flips]))

        # TD Update
        td_target = reward + (self.config.discount_factor_gamma * expected_future_q)
        self.q_table[state][action] += self.config.learning_rate_alpha * (td_target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(
            self.config.minimum_epsilon,
            self.epsilon * self.config.epsilon_decay_rate
        )
