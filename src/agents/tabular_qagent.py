from collections import defaultdict
from typing import Tuple, Any, Dict
import numpy as np
from src.core.agent import BaseAgent
from src.core.config import AgentConfig

class TabularQAgent(BaseAgent):
    """
    HD-Compliant Tabular Q-Learning Agent.
    Implements Sarah's 'Expected Value' update by empirically estimating 
    the lake transition probability P.
    """

    def __init__(self, agent_id: str, config: AgentConfig, shared_q_table: Dict = None):
        super().__init__(agent_id, config)
        
        # HD Requirement: Shared Policies
        if shared_q_table is not None:
            self.q_table = shared_q_table
        else:
            self.q_table = defaultdict(lambda: np.zeros(self.config.action_size))

        self.epsilon = self.config.initial_epsilon
        
        # Empirical Transition Estimation
        # We track how many times the lake changed state vs. total steps observed
        self.transition_counts = 0
        self.total_observations = 0
        self.p_hat_flip = 0.5 # Initial prior before data

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
        # 1. Update Empirical Transition Estimate
        # State: (y, x, has_payload, lake_flooded)
        current_lake = state[3]
        observed_next_lake = next_obs_state[3]
        
        self.total_observations += 1
        if current_lake != observed_next_lake:
            self.transition_counts += 1
        
        # Update p_hat (Probability that the lake FLIPS state)
        self.p_hat_flip = self.transition_counts / self.total_observations
        
        current_q = self.q_table[state][action]
        
        if terminal:
            expected_future_q = 0.0
        else:
            y_next, x_next, payload_next, _ = next_obs_state
            p = self.p_hat_flip
            
            # The two possible future universes for the lake state
            state_stays = (y_next, x_next, payload_next, current_lake)
            state_flips = (y_next, x_next, payload_next, not current_lake)
            
            max_q_stays = np.max(self.q_table[state_stays])
            max_q_flips = np.max(self.q_table[state_flips])
            
            # Expected Future Value Equation:
            # E[max Q'] = P(stay) * max Q(S'_stay) + P(flip) * max Q(S'_flip)
            expected_future_q = ( (1 - p) * max_q_stays ) + ( p * max_q_flips )

        # TD Update
        td_target = reward + (self.config.discount_factor_gamma * expected_future_q)
        self.q_table[state][action] += self.config.learning_rate_alpha * (td_target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(
            self.config.minimum_epsilon,
            self.epsilon * self.config.epsilon_decay_rate
        )
