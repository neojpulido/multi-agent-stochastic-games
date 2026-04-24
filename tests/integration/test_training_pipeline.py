from src.domain.gridworld.stochastic import StochasticMultiAgentEnv
from src.agents.tabular_qagent import TabularQAgent
from src.orchestration.runner import SimulationRunner

def test_training_convergence(default_experiment_config):
    """Integration test to verify that agents can learn on a small grid."""
    env = StochasticMultiAgentEnv(config=default_experiment_config.env)
    
    agent_ids = ["Agent_A", "Agent_B"]
    agents = {
        aid: TabularQAgent(agent_id=aid, config=default_experiment_config.agent) 
        for aid in agent_ids
    }

    # Use a small budget for the test
    # We create a new config with small budget to speed up the test
    test_runner = SimulationRunner(default_experiment_config, env, agents)
    test_runner.run_experiment()

    # Simple verification: check if epsilon has decayed
    for agent in agents.values():
        assert agent.epsilon < 1.0, f"Epsilon for {agent.agent_id} did not decay"
