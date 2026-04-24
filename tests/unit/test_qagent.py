import numpy as np
from qlearning import TabularQAgent

def test_agent_initialization(default_agent_config):
    agent = TabularQAgent(config=default_agent_config)
    assert agent.config.action_size == 8
    assert agent.current_exploration_probability == 1.0

def test_select_action_greedy(default_agent_config):
    agent = TabularQAgent(config=default_agent_config)
    state = ((0,0), (1,1), False)
    # Set a specific Q-value
    agent.action_value_table[state][3] = 10.0
    action = agent.select_action(state, use_greedy=True)
    assert action == 3

def test_agent_learn_step(default_agent_config):
    """
    Rigorously verifies the Bellman Optimality Equation:
    Q(s, a) ← Q(s, a) + alpha * [Reward + gamma * max_a' Q(s', a') - Q(s, a)]
    """
    agent = TabularQAgent(config=default_agent_config)
    previous_state = ((0,0), (1,1), False)
    current_state = ((0,1), (1,1), False)
    action = 1
    reward = 10.0
    
    # Setup controlled Q-values for the future state
    # Set max_a' Q(s', a') to 5.0
    agent.action_value_table[current_state][3] = 5.0
    
    # Retrieve hyperparameters
    alpha = agent.config.learning_rate_alpha  # Default 0.1
    gamma = agent.config.discount_factor_gamma  # Default 0.99
    
    # 1. INITIAL Q(s, a) is 0.0 (by defaultdict)
    initial_q = agent.action_value_table[previous_state][action]
    assert initial_q == 0.0
    
    # 2. PERFORM LEARNING STEP
    agent.apply_learning_step(previous_state, action, reward, current_state, is_terminal_state=False)
    
    # 3. VERIFY MATHEMATICAL UPDATE
    # Expected: 0 + 0.1 * [10.0 + 0.99 * 5.0 - 0] 
    # = 0.1 * [10.0 + 4.95] = 1.495
    expected_q = initial_q + alpha * (reward + gamma * 5.0 - initial_q)
    updated_q = agent.action_value_table[previous_state][action]
    
    assert np.isclose(updated_q, expected_q), f"Q-Update failed! Expected {expected_q}, got {updated_q}"
    
    # 4. VERIFY TERMINAL STATE MATH
    # If the state is terminal, the future value should be ignored (0)
    # Reset Q-table for this test
    agent.action_value_table[previous_state][action] = 0.0
    agent.apply_learning_step(previous_state, action, reward, current_state, is_terminal_state=True)
    
    # Expected: 0 + 0.1 * [10.0 + 0.99 * 0 - 0] = 1.0
    expected_terminal_q = 0 + alpha * (reward + 0 - 0)
    assert np.isclose(agent.action_value_table[previous_state][action], expected_terminal_q)
