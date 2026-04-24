from collections import defaultdict
from typing import Tuple, Any
import numpy as np
from qlearning.core.agent import BaseAgent
from qlearning.core.config import AgentConfig

# =============================================================================
# AGENT HYPERPARAMETERS (DEFAULTS)
# =============================================================================
# alpha (α): Step size for Q-table updates
DEFAULT_LEARNING_RATE = 0.1
# gamma (γ): Weight of future rewards (Discount Factor)
DEFAULT_DISCOUNT_FACTOR = 0.99
# epsilon (ε): Initial probability of choosing a random action
DEFAULT_INITIAL_EXPLORATION = 1.0


class TabularQAgent(BaseAgent):
    """
    Agent implementation using the Tabular Q-Learning algorithm.

    This agent learns an optimal policy by estimating the state-action value
    function Q(s, a) through interaction with the environment. It uses
    Temporal Difference (TD) learning to update its knowledge.
    """

    def __init__(self, config: AgentConfig = AgentConfig()):
        """
        Initializes the agent with a centralized configuration.

        Args:
            config: AgentConfig object containing hyperparameters.
        """
        self.config = config

        # Q-Table Initialization: Q(s, a)
        # We use a sparse mapping (defaultdict) where every unseen state
        # is initialized to a vector of zeros representing all possible actions.
        self.action_value_table = defaultdict(lambda: np.zeros(self.config.action_size))

        # Exploration rate (epsilon) for the epsilon-greedy strategy.
        self.current_exploration_probability = self.config.initial_epsilon

    def select_action(self, current_state: Tuple[Any, ...], use_greedy: bool = False) -> int:
        """
        PHASE: ACTION SELECTION (Epsilon-Greedy Policy)

        Mathematically:
        π(s) = { random action from A         with probability ε
               { argmax_a Q(s, a)             with probability 1 - ε

        Args:
            current_state: The state the agent is currently in (s_t).
            use_greedy: If True, the agent exploits only (for evaluation).
        """
        # 1. EXPLORATION: Uniformly random action selection
        if not use_greedy and np.random.rand() < self.current_exploration_probability:
            return int(np.random.randint(self.config.action_size))

        # 2. EXPLOITATION: Optimal action selection based on current knowledge
        # Vectorized argmax over the action-value array for the current state.
        return int(np.argmax(self.action_value_table[current_state]))

    def apply_learning_step(self,
                            previous_state: Tuple[Any, ...],
                            action: int,
                            reward: float,
                            current_state: Tuple[Any, ...],
                            is_terminal_state: bool) -> None:
        """
        PHASE: LEARNING (Bellman Optimality Update)

        This method implements the core Temporal Difference (TD) learning
        using the Tabular Q-Learning update rule:

        Q(s, a) ← Q(s, a) + α * [R + γ * max_a' Q(s', a') - Q(s, a)]

        Where:
            α (alpha) is the learning rate.
            γ (gamma) is the discount factor.
            R is the reward received after taking action a in state s.
            max_a' Q(s', a') is the maximum Q-value for the next state s'.

        Args:
            previous_state (s): The state before the action was taken (s_t).
            action (a): The action executed (a_t).
            reward (R): The feedback signal received (R_{t+1}).
            current_state (s'): The resulting state after the action (s_{t+1}).
            is_terminal_state: Boolean indicating if the transition led to a goal.
        """
        # 1. RETRIEVE CURRENT ESTIMATE Q(s, a)
        # Access the Q-table for the previous state and specific action taken.
        current_q_value = self.action_value_table[previous_state][action]

        # 2. COMPUTE ESTIMATED OPTIMAL FUTURE VALUE max_a' Q(s', a')
        # If the next state is terminal, the future reward is zero by definition.
        if is_terminal_state:
            max_future_q = 0.0
        else:
            # We look ahead at the resulting 'current_state' to find the best possible action-value.
            max_future_q = np.max(self.action_value_table[current_state])

        # 3. COMPUTE THE TD TARGET: R + γ * max_a' Q(s', a')
        td_target = reward + (self.config.discount_factor_gamma * max_future_q)

        # 4. COMPUTE THE TD ERROR (Temporal Difference): Target - current estimate
        td_error = td_target - current_q_value

        # 5. UPDATE Q-TABLE: Q(s,a) ← Q(s,a) + α * TD_Error
        # This is the foundational Bellman update for off-policy Q-Learning.
        self.action_value_table[previous_state][action] += self.config.learning_rate_alpha * td_error

        # 6. ANNEAL EXPLORATION RATE (Epsilon Decay)
        # We decrease the probability of random actions as the agent gains more knowledge.
        if is_terminal_state:
            self.current_exploration_probability = max(
                self.config.minimum_epsilon,
                self.current_exploration_probability * self.config.epsilon_decay_rate
            )
