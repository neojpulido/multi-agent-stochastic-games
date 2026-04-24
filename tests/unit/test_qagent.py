import numpy as np
from src.agents.tabular_qagent import TabularQAgent

def test_agent_initialization(default_agent_config):
    agent = TabularQAgent(agent_id="TestAgent", config=default_agent_config)
    assert agent.config.action_size == 5
    assert agent.epsilon == 1.0

def test_choose_action_greedy(default_agent_config):
    agent = TabularQAgent(agent_id="TestAgent", config=default_agent_config)
    state = (2, 2, False, False) # pos_x, pos_y, has_sample, lake_flooded
    # Set a specific Q-value
    agent.q_table[state][3] = 10.0
    action = agent.choose_action(state) # Non-deterministic if epsilon=1, but let's test logic
    # Set epsilon to 0 to force greedy
    agent.epsilon = 0.0
    action = agent.choose_action(state)
    assert action == 3

def test_agent_learn_step(default_agent_config):
    agent = TabularQAgent(agent_id="TestAgent", config=default_agent_config)
    previous_state = (0, 0, False, False)
    current_state = (0, 1, False, False)
    action = 1
    reward = 10.0
    
    # Setup controlled Q-values for the future state
    agent.q_table[current_state][3] = 5.0
    
    alpha = agent.config.learning_rate_alpha  # 0.1
    gamma = agent.config.discount_factor_gamma  # 0.9
    
    initial_q = agent.q_table[previous_state][action] # 0.0
    
    agent.update_learning(previous_state, action, reward, current_state, terminal=False)
    
    # Expected: 0 + 0.1 * [10.0 + 0.9 * 5.0 - 0] = 1.45
    expected_q = initial_q + alpha * (reward + gamma * 5.0 - initial_q)
    updated_q = agent.q_table[previous_state][action]
    
    assert np.isclose(updated_q, expected_q)
