import os
from dataclasses import replace
from src.core.config import ExperimentConfig
from src.domain.gridworld.stochastic import StochasticMultiAgentEnv
from src.agents.tabular_qagent import TabularQAgent
from src.orchestration.runner import SimulationRunner

def run_hd_experiment(budget: int = 2000):
    """
    HD Experiment: Systematically modify the step penalty (delay cost) in Phase 2
    to identify the tipping point where coordination fails.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base_config_path = os.path.join(project_root, "configs", "stage2_robert_efficient.json")
    
    if not os.path.exists(base_config_path):
        print(f"Error: Base config {base_config_path} not found.")
        return

    print("\n=== Running HD Experiment: Phase 2 Step Cost Tipping Point ===")
    print(f"Budget per condition: {budget} episodes.")
    
    step_costs = [-1.0, -3.0, -5.0, -7.0, -10.0, -15.0]
    
    for cost in step_costs:
        # 1. Configuration
        base_config = ExperimentConfig.from_json(base_config_path)
        new_env_config = replace(base_config.env, step_cost=cost, hazard_penalty=0.0)
        config = replace(base_config, env=new_env_config, training_episode_budget=budget)
        
        # 2. Environment & Agents
        env = StochasticMultiAgentEnv(config.env)
        agents = {aid: TabularQAgent(aid, config.agent) for aid in ["Agent_A", "Agent_B"]}
        
        # 3. Execution
        # We run the simulation and extract the final evaluation metrics
        runner = SimulationRunner(config, env, agents)
        history = runner.run_experiment(eval_interval=budget) # Only eval at the very end
        
        final_collision_rate = history["collision_rate"][-1]
        print(f"Step Cost: {cost:>5.1f} | Final Avg Collision Rate: {final_collision_rate:5.2f}")

if __name__ == "__main__":
    run_hd_experiment()
