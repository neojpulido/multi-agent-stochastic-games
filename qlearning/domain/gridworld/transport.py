from typing import Tuple, Any, Optional
import numpy as np
from qlearning.core.reward import BaseReward
from qlearning.core.config import EnvConfig
from qlearning.domain.rewards.transport import TransportReward
from qlearning.core.actions import Actions

# =============================================================================
# ENVIRONMENT CONSTANTS (DEFAULTS)
# =============================================================================
DEFAULT_GRID_ROWS = 5            # Height of the grid (M)
DEFAULT_GRID_COLS = 5            # Width of the grid (N)
DEFAULT_STEP_COST = -1.0         # Penalty per step to encourage efficiency
DEFAULT_SUCCESS_REWARD = 100.0   # Reward for successful delivery to Goal B

class GridWorldTransport:
    """
    Rectangular grid-based environment for pickup and delivery transport tasks.
    
    This class models the environment dynamics, state transitions, and 
    physical constraints of the world. It provides states to the agent
    and executes actions to transition between states.
    """
    def __init__(self, config: EnvConfig = EnvConfig(), reward_provider: Optional[BaseReward] = None):
        """
        Initializes the world with generic MxN dimensions.
        
        Args:
            config: EnvConfig object containing dimensions and reward values.
            reward_provider: Concrete implementation of BaseReward.
        """
        self.config = config
        
        # Fixed Goal B is at the bottom-right corner: (M-1, N-1)
        self.goal_coordinates = (self.config.grid_rows - 1, self.config.grid_cols - 1)
        
        # Dependency Injection for Reward Logic
        self.reward_provider = reward_provider or TransportReward(config=self.config)
        self.reset()

    def reset(self) -> Tuple[Any, ...]:
        """
        PHASE: INITIALIZATION
        Resets the world to a new starting configuration for a fresh episode.
        """
        # Random starting coordinates for the agent
        self.agent_coordinates = (np.random.randint(self.config.grid_rows), np.random.randint(self.config.grid_cols))
        
        # Random starting coordinates for the item (Pickup A)
        # Ensure the item does not spawn directly on the goal.
        while True:
            self.pickup_coordinates = (np.random.randint(self.config.grid_rows), np.random.randint(self.config.grid_cols))
            if self.pickup_coordinates != self.goal_coordinates:
                break
                
        # Status tracker: whether the agent is currently carrying the payload.
        self.has_payload = False
        
        return self.get_state()

    def get_state(self) -> Tuple[Any, ...]:
        """
        PHASE: PERCEPTION
        Constructs the state representation provided to the agent (S_t).
        
        Returns:
            A tuple of (agent_pos, pickup_pos, has_payload)
        """
        return (self.agent_coordinates, self.pickup_coordinates, self.has_payload)

    def execute_step(self, action_idx: int) -> Tuple[Tuple[Any, ...], float, bool]:
        """
        PHASE: DYNAMICS (State Transition)
        Executes an action and calculates the resulting environment state.
        
        Args:
            action_idx: Index representing the direction of movement.
            
        Returns:
            next_state: The new state (S_{t+1})
            reward: The scalar feedback signal (R_{t+1})
            is_terminal_state: Whether the goal was reached.
        """
        # 1. Store the previous state for reward calculation
        state_before_action = self.get_state()

        # 2. Transition Logic: Compute next potential position
        potential_next_position = Actions.calculate_next_coordinates(self.agent_coordinates, action_idx)

        # 3. Validation: Check if the move is legal (within grid boundaries)
        if Actions.is_within_boundaries(potential_next_position, self.config.grid_rows, self.config.grid_cols):
            self.agent_coordinates = potential_next_position

        # 4. Mechanics: Automatic pickup upon entering the pickup coordinates
        if not self.has_payload and self.agent_coordinates == self.pickup_coordinates:
            self.has_payload = True

        # 5. Mechanics: Check for termination (delivery at goal B)
        is_terminal_state = False
        if self.has_payload and self.agent_coordinates == self.goal_coordinates:
            is_terminal_state = True

        # 6. Capture resulting state
        next_state = self.get_state()

        # 7. REWARD: Compute feedback signal using the injected reward provider
        reward = self.reward_provider.calculate_reward(state_before_action, action_idx, next_state, is_terminal_state)

        return next_state, reward, is_terminal_state
