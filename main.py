import argparse
import os
from src.core.config import ExperimentConfig
from src.domain.gridworld.stochastic import StochasticMultiAgentEnv
from src.agents.tabular_qagent import TabularQAgent
from src.orchestration.runner import SimulationRunner

def main():
    parser = argparse.ArgumentParser(description="MARL Stochastic Games Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage2_sarah_safe.json",
        help="Path to the experiment JSON configuration."
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found.")
        return

    config = ExperimentConfig.from_json(args.config)
    
    # 1. Initialize Environment
    # We exclusively support the Stage 2 Stochastic Multi-Agent environment.
    env = StochasticMultiAgentEnv(config.env)
    agent_ids = ["Agent_A", "Agent_B"]

    # 2. Initialize Agents
    agents = {
        aid: TabularQAgent(agent_id=aid, config=config.agent) 
        for aid in agent_ids
    }

    # 3. Run Orchestrator
    runner = SimulationRunner(config, env, agents)
    runner.run_experiment()

if __name__ == "__main__":
    main()
