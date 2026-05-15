import argparse
import os
from src.core.config import ExperimentConfig
from src.domain.gridworld.stochastic import StochasticMultiAgentEnv
from src.agents.tabular_qagent import TabularQAgent
from src.orchestration.runner import SimulationRunner
from src.orchestration.jupyter_utils import JupyterEvaluator
from src.orchestration.hd_experiment import run_hd_experiment
from src.math.egt_model import run_simulation as run_egt_simulation

def run_phase(config_path: str):
    """
    Standard workflow for running a MARL phase (Audit, Train, Evaluate).
    """
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found.")
        return

    config = ExperimentConfig.from_json(config_path)
    
    # 1. Initialize Environment
    env = StochasticMultiAgentEnv(config.env)
    agent_ids = ["Agent_A", "Agent_B"]

    # 2. Initialize Agents
    agents = {
        aid: TabularQAgent(agent_id=aid, config=config.agent) 
        for aid in agent_ids
    }

    # 3. Run Orchestrator
    runner = SimulationRunner(config, env, agents)
    history = runner.run_experiment()

    # 4. Visualization & Learning Curves (Module logic)
    # Note: Learning curves require plt.show()
    JupyterEvaluator.plot_learning_curves(history)
    
    print(f"\n--- Final Policy Visualization for {config.experiment_name} ---")
    JupyterEvaluator.visualize_policy(agents["Agent_A"], env, lake_flooded=True, has_sample=True)
    JupyterEvaluator.visualize_policy(agents["Agent_B"], env, lake_flooded=True, has_sample=False)

def main():
    parser = argparse.ArgumentParser(description="FIT5226 MARL Project Architect Toolkit")
    parser.add_argument(
        "mode",
        choices=["task1", "task2", "experiment", "egt"],
        help="Mode to run: task1 (Phase 1), task2 (Phase 2), experiment (HD Step Cost), egt (Replicator Dynamics)."
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Override training budget (episodes)."
    )
    args = parser.parse_args()

    if args.mode == "task1":
        run_phase("configs/stage2_sarah_safe.json")
    elif args.mode == "task2":
        run_phase("configs/stage2_robert_efficient.json")
    elif args.mode == "experiment":
        budget = args.budget if args.budget else 2000
        run_hd_experiment(budget=budget)
    elif args.mode == "egt":
        run_egt_simulation()

if __name__ == "__main__":
    main()
