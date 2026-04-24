import os
import time
import numpy as np

# Package is in root, no sys.path hacking needed for imports to work.
from qlearning import ExperimentConfig, TransportGridWorld, TabularQAgent

# =============================================================================
# ASSIGNMENT CONSTRAINTS & METRICS (Source of Truth for Rubric)
# =============================================================================
ALLOWED_TRAINING_BUDGET = 20000  # Strict upper limit for episodes
TARGET_SUCCESS_RATE = 100.0      # Required percentage for outstanding grade
MAX_ALLOWED_STEPS = 10           # Efficiency limit for any scenario
DEFAULT_GRID_DIMENSIONS = (5, 5) # Standard M x N size

def run_evaluation():
    """
    PERFORMANCE EVALUATION SUITE
    
    This script automates the training and exhaustive verification of the agent.
    It ensures the policy learned is both robust (100% success) and 
    optimal (minimum steps).
    """
    # 1. LOAD CONFIGURATION
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(PROJECT_ROOT, 'configs', 'default_experiment.json')
    
    if os.path.exists(config_path):
        config = ExperimentConfig.initialize_from_json(config_path)
        print(f"[*] Configuration loaded from: {config_path}")
    else:
        print(f"[!] Warning: Config not found. Using system defaults.")
        config = ExperimentConfig()

    # 2. INITIALIZE ENVIRONMENT & AGENT
    env = TransportGridWorld(config=config.env)
    agent = TabularQAgent(config=config.agent)

    # 3. TRAINING PHASE
    print(f"[*] Starting Training: {config.training_episode_budget} episodes...")
    t_start = time.process_time()
    
    for episode in range(1, config.training_episode_budget + 1):
        previous_state = env.reset()
        done = False
        
        # Training loop: Perception -> Action -> Learning
        while not done:
            action = agent.select_action(previous_state)
            current_state, reward, done = env.execute_step(action)
            agent.apply_learning_step(previous_state, action, reward, current_state, done)
            previous_state = current_state
        
        if episode % 5000 == 0:
            print(f"    > Progress: {episode} episodes completed.")

    train_time = time.process_time() - t_start
    print(f"[+] Training finished in {train_time:.2f} seconds.")

    # 4. EXHAUSTIVE EVALUATION PHASE
    print("\n[*] Commencing Exhaustive Evaluation (All possible scenarios)...")
    rows = env.config.grid_rows
    cols = env.config.grid_cols
    goal = env.goal_coordinates
    
    success_count = 0
    total_scenarios = 0
    max_steps_observed = 0

    # Combinatoric Iteration: Every valid (Start, Pickup) coordinate pair
    for start_row in range(rows):
        for start_col in range(cols):
            for pickup_row in range(rows):
                for pickup_col in range(cols):
                    # Exclude trivial cases where start or pickup is on the goal
                    if (start_row, start_col) == goal or (pickup_row, pickup_col) == goal:
                        continue
                    
                    # Manually inject state for verification
                    env.agent_coordinates = (start_row, start_col)
                    env.pickup_coordinates = (pickup_row, pickup_col)
                    env.has_payload = False
                    state = env.get_state()
                    
                    done = False
                    steps = 0
                    
                    # Pure Exploitation: test the learned policy
                    while not done and steps < 20:
                        action = agent.select_action(state, use_greedy=True)
                        state, _, done = env.execute_step(action)
                        steps += 1
                    
                    total_scenarios += 1
                    if done and steps <= MAX_ALLOWED_STEPS:
                        success_count += 1
                    max_steps_observed = max(max_steps_observed, steps)

    success_rate = (success_count / total_scenarios) * 100
    
    # 5. FINAL REPORTING
    print("-" * 50)
    print(f"{'METRIC':<30} | {'VALUE':<15}")
    print("-" * 50)
    print(f"{'Total Unique Scenarios':<30} | {total_scenarios:<15}")
    print(f"{'Success Rate (<= 10 steps)':<30} | {success_rate:>14.2f}%")
    print(f"{'Maximum Steps Taken':<30} | {max_steps_observed:<15}")
    print("-" * 50)

    if success_rate == TARGET_SUCCESS_RATE and max_steps_observed <= MAX_ALLOWED_STEPS:
        print("\n✅ PASSED: All FIT5226 assignment requirements met!")
    else:
        print("\n❌ FAILED: Performance constraints violated.")

if __name__ == "__main__":
    run_evaluation()
