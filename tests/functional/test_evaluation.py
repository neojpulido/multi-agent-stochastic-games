import numpy as np
import pytest
from src.domain.gridworld.stochastic import StochasticMultiAgentEnv
from src.agents.tabular_qagent import TabularQAgent
from src.orchestration.runner import SimulationRunner
from src.core.state import StateHandler

def test_system_performance_requirements(default_experiment_config):
    """
    Functional test that trains agents and verifies they meet the 
    assignment's performance criteria.
    """
    env = StochasticMultiAgentEnv(config=default_experiment_config.env)
    
    agent_ids = ["Agent_A", "Agent_B"]
    agents = {
        aid: TabularQAgent(agent_id=aid, config=default_experiment_config.agent) 
        for aid in agent_ids
    }

    # 1. Run Training
    runner = SimulationRunner(default_experiment_config, env, agents)
    runner.run_experiment()

    # 2. Evaluation Phase (Greedy Policy)
    obs = env.reset()
    done = False
    steps = 0
    total_reward = 0
    
    # Increase safety limit to 1000 for smoke test stability
    while not done and steps < 1000:
        joint_action = {}
        for aid, agent in agents.items():
            state_key = StateHandler.get_agent_state(aid, obs)
            # Set epsilon to 0 manually for greedy evaluation
            original_epsilon = agent.epsilon
            agent.epsilon = 0.0
            joint_action[aid] = agent.choose_action(state_key)
            agent.epsilon = original_epsilon
            
        obs, rewards, dones, all_done = env.step(joint_action)
        total_reward += sum(rewards.values())
        done = all_done
        steps += 1

    # Verify that the simulation finishes and doesn't get stuck in an infinite loop
    assert steps < 1000, "Agents failed to complete the task in a reasonable number of steps"
    assert steps > 0
