import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from src.core.state import StateHandler
from src.core.actions import Actions, Directions

class JupyterEvaluator:
    """
    Jupyter-compatible tools for evaluating and visualizing MARL agents.
    Provides rigorous metrics and quiver-plot policy visualization for Task 1.
    """

    @staticmethod
    def evaluate_greedy(env: Any, agents: Dict[str, Any], episodes: int = 20) -> Dict[str, float]:
        """
        Runs a pure greedy evaluation loop (epsilon=0).
        Returns averaged metrics for performance monitoring.
        """
        metrics = {
            "avg_return": 0.0,
            "avg_length": 0.0,
            "collision_rate": 0.0,
            "hazard_rate": 0.0
        }
        
        for _ in range(episodes):
            obs = env.reset()
            dones = {aid: False for aid in agents.keys()}
            episode_return = 0.0
            steps = 0
            
            while not all(dones.values()) and steps < 500:
                joint_action = {}
                for aid, agent in agents.items():
                    if not dones[aid]:
                        state_key = StateHandler.get_agent_state(aid, obs)
                        joint_action[aid] = agent.choose_action(state_key, greedy=True)
                    else:
                        joint_action[aid] = 4 # Directions.WAIT
                        
                next_obs, rewards, dones_next, _ = env.step(joint_action)
                
                episode_return += sum(rewards.values())
                
                # Check for physical violations (active agents only)
                if not dones["Agent_A"] and not dones["Agent_B"]:
                     if env.positions["Agent_A"] == env.positions["Agent_B"]:
                         metrics["collision_rate"] += 1
                
                if not dones["Agent_A"] and env.lake_flooded and env.positions["Agent_A"] == env.locs["Lake"]:
                     metrics["hazard_rate"] += 1
                     
                obs = next_obs
                dones = dones_next
                steps += 1
                
            metrics["avg_return"] += episode_return
            metrics["avg_length"] += steps
            
        # Average results
        for k in metrics.keys():
            metrics[k] /= episodes
            
        return metrics

    @staticmethod
    def visualize_policy(agent: Any, env: Any, lake_flooded: bool, has_sample: bool):
        """
        Renders a high-resolution quiver plot showing the preferred action in each state.
        Highlights the learned 'Traffic Light' behavior.
        """
        rows, cols = env.grid_size
        U, V = np.zeros((rows, cols)), np.zeros((rows, cols))
        
        # Colors/Markers for points of interest
        target_loc = env.locs["U"] if agent.agent_id == "Agent_A" else env.locs["V"]
        start_loc = env.locs["X"] if agent.agent_id == "Agent_A" else env.locs["Y"]
        lake_loc = env.locs["Lake"]

        for r in range(rows):
            for c in range(cols):
                state = (r, c, has_sample, lake_flooded)
                # Check if state was explored (non-zero entries in Q-table)
                q_vals = agent.q_table[state]
                if np.max(q_vals) != 0 or np.min(q_vals) != 0:
                    action = np.argmax(q_vals)
                    dy, dx = Actions.get_delta(action)
                    U[r, c], V[r, c] = dx, dy
                else:
                    U[r, c], V[r, c] = 0, 0 # Unexplored

        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create grid background
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Plot quiver
        # Pivot='mid' centers arrows on cells
        q = ax.quiver(np.arange(cols), np.arange(rows), U, V, color='teal', 
                      pivot='mid', scale=15, width=0.005, headwidth=5)
        
        # Mark key locations
        ax.plot(start_loc[1], start_loc[0], 'rs', markersize=12, label='Start/End')
        ax.plot(target_loc[1], target_loc[0], 'go', markersize=12, label='Target (Pickup)')
        
        # Lake visualization
        lake_color = 'blue' if lake_flooded else 'navajowhite'
        ax.add_patch(plt.Rectangle((lake_loc[1]-0.5, lake_loc[0]-0.5), 1, 1, 
                                   color=lake_color, alpha=0.3, label='Lake'))
        
        title = f"Policy: {agent.agent_id} | Lake: {'FLOODED' if lake_flooded else 'DRY'} | Payload: {has_sample}"
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.invert_yaxis() # Match grid coordinate system (0,0 at top-left)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_learning_curves(history: Dict[str, List[float]], window: int = 5):
        """
        Improved visual demonstration of MARL convergence.
        Plots moving averages for stability with raw evaluation points in background.
        """
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        episodes = history["episodes"]
        
        def moving_average(data, w):
            if len(data) < w: return data
            return np.convolve(data, np.ones(w)/w, mode='same')

        # 1. Average Return
        axs[0].plot(episodes, history["avg_return"], alpha=0.2, color='gray', label="Raw")
        if len(episodes) >= window:
            axs[0].plot(episodes, moving_average(history["avg_return"], window), color='blue', linewidth=2, label=f"MA(w={window})")
        axs[0].set_title("Evaluation Return", fontsize=14)
        axs[0].set_xlabel("Episode")
        axs[0].set_ylabel("Total Return")
        axs[0].grid(True, linestyle=':', alpha=0.6)
        axs[0].legend()

        # 2. Episode Length
        axs[1].plot(episodes, history["avg_length"], alpha=0.2, color='gray', label="Raw")
        if len(episodes) >= window:
            axs[1].plot(episodes, moving_average(history["avg_length"], window), color='green', linewidth=2, label=f"MA(w={window})")
        axs[1].set_title("Episode Length", fontsize=14)
        axs[1].set_xlabel("Episode")
        axs[1].set_ylabel("Steps")
        axs[1].grid(True, linestyle=':', alpha=0.6)
        axs[1].legend()

        # 3. Physical Violations
        axs[2].plot(episodes, history["collision_rate"], alpha=0.2, color='red', linestyle='--')
        axs[2].plot(episodes, history["hazard_rate"], alpha=0.2, color='orange', linestyle='--')
        if len(episodes) >= window:
            axs[2].plot(episodes, moving_average(history["collision_rate"], window), color='red', linewidth=2, label="Collisions")
            axs[2].plot(episodes, moving_average(history["hazard_rate"], window), color='orange', linewidth=2, label="Water Hazards")
        axs[2].set_title("Violation Rates", fontsize=14)
        axs[2].set_xlabel("Episode")
        axs[2].set_ylabel("Occurrences")
        axs[2].grid(True, linestyle=':', alpha=0.6)
        axs[2].legend()

        plt.suptitle("Task 1: Sarah's Safe Behavior Convergence", fontsize=16, y=1.05)
        plt.tight_layout()
        plt.show()
