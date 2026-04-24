import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Any, Dict
from qlearning.domain.gridworld.transport import TransportGridWorld

class GridVisualiser:
    """Provides high-quality visualisations for the GridWorld environment."""
    
    @staticmethod
    def plot_policy_path(env: TransportGridWorld, path: List[Tuple[int, int]], title: str = "Agent Path"):
        """
        Renders the grid with the agent's path, source, and destination.
        
        Args:
            env: The TransportGridWorld environment instance.
            path: List of (row, col) tuples representing the agent's journey.
            title: Title for the plot.
        """
        rows = env.config.grid_rows
        cols = env.config.grid_cols
        
        fig, ax = plt.subplots(figsize=(max(cols, 6), max(rows, 6)))
        
        # Draw grid lines
        ax.set_xticks(np.arange(-.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
        
        # Plot markers
        # Source (Pickup A) - Green Circle
        ax.plot(env.pickup_coordinates[1], env.pickup_coordinates[0], 'go', markersize=20, label='Pickup A', alpha=0.6)
        # Destination (Dropoff B) - Red Square
        ax.plot(env.goal_coordinates[1], env.goal_coordinates[0], 'rs', markersize=20, label='Dropoff B', alpha=0.6)
        
        # Plot path
        if path:
            # path is a list of (row, col)
            y_coords, x_coords = zip(*path)
            ax.plot(x_coords, y_coords, 'b--', linewidth=3, label='Agent Path', alpha=0.8)
            # Start marker - Blue Star
            ax.plot(x_coords[0], y_coords[0], 'b*', markersize=15, label='Start')
            # End marker - marker at the last position
            ax.plot(x_coords[-1], y_coords[-1], 'ro', markersize=10)
        
        ax.set_title(title, fontsize=16)
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5) # Invert Y for matrix coordinates (0,0 at top-left)
        
        # Add labels to grid cells
        for r in range(rows):
            for c in range(cols):
                ax.text(c, r, f"({r},{c})", va='center', ha='center', color='gray', fontsize=8)

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_learning_curve(rewards: List[float], window: int = 100, title: str = "Learning Curve"):
        """
        Plots the moving average of rewards during training.
        
        Args:
            rewards: List of total rewards per episode.
            window: Moving average window size.
            title: Title for the plot.
        """
        plt.figure(figsize=(12, 6))
        
        if len(rewards) >= window:
            averages = np.convolve(rewards, np.ones(window)/window, mode='valid')
            plt.plot(averages, label=f"Moving Avg (w={window})", color='blue', linewidth=2)
        
        plt.plot(rewards, alpha=0.2, color='gray', label="Raw Rewards")
        
        plt.title(title, fontsize=16)
        plt.xlabel("Episode", fontsize=12)
        plt.ylabel("Total Reward", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.show()

    @staticmethod
    def plot_performance_comparison(results: Dict[str, List[float]], window: int = 500, title: str = "Performance Comparison"):
        """
        Compares the learning progress of multiple agents or configurations.
        
        Args:
            results: Dictionary mapping label names to lists of rewards.
            window: Moving average window size.
            title: Title for the comparison plot.
        """
        plt.figure(figsize=(14, 7))
        
        for label, rewards in results.items():
            if len(rewards) >= window:
                averages = np.convolve(rewards, np.ones(window)/window, mode='valid')
                plt.plot(averages, label=label, linewidth=2)
            else:
                plt.plot(rewards, label=label, alpha=0.5)
                
        plt.title(title, fontsize=18)
        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Average Reward (Moving Window)", fontsize=14)
        plt.grid(True, which='both', linestyle=':', alpha=0.8)
        plt.legend(fontsize=12)
        plt.show()

    @staticmethod
    def visualise_policy_grid(env: TransportGridWorld, agent: Any):
        """
        Renders a quiver plot showing the preferred action in each state.
        Only visualises the grid where has_payload=False for simplicity.
        """
        rows = env.config.grid_rows
        cols = env.config.grid_cols
        
        U = np.zeros((rows, cols))
        V = np.zeros((rows, cols))
        
        # Directions mapping to (dx, dy)
        from qlearning.core.actions import Actions, Directions
        
        for r in range(rows):
            for c in range(cols):
                state = ((r, c), env.pickup_coordinates, False)
                action = agent.select_action(state, use_greedy=True)
                direction = Directions(action)
                dx, dy = Actions.map_direction_to_unit_vector(direction)
                U[r, c] = dy # Plotting library uses (x, y) where y is up
                V[r, c] = -dx # Matrix row increases downward, so dx needs flip for y-axis
                
        fig, ax = plt.subplots(figsize=(8, 8))
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        ax.quiver(x, y, U, V, color='teal', pivot='mid', scale=20)
        
        ax.set_xticks(np.arange(-.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
        
        # Plot Pickup A
        ax.plot(env.pickup_coordinates[1], env.pickup_coordinates[0], 'go', markersize=15, label='Pickup A', alpha=0.4)
        
        ax.set_title("Learned Policy (Greedy Actions)", fontsize=16)
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)
        plt.show()
