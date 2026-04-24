from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json


@dataclass(frozen=True)
class AgentConfig:
    """Hyperparameters for the Tabular Q-Learning Agent."""
    learning_rate_alpha: float
    discount_factor_gamma: float
    initial_epsilon: float
    epsilon_decay_rate: float
    minimum_epsilon: float
    action_size: int  # 8 for Assignment 1, 9 (inc. WAIT) for Assignment 2


@dataclass(frozen=True)
class EnvConfig:
    """Structural parameters and reward values for the GridWorld."""
    grid_rows: int
    grid_cols: int
    p_flood: float  # Probability of lake flooding (Stochasticity)
    step_cost: float
    success_reward: float
    collision_penalty: float = 0.0  # Relevant for Stage 2
    hazard_penalty: float = 0.0  # Sarah's safety penalty


@dataclass(frozen=True)
class ExperimentConfig:
    """The root configuration object for a specific simulation run."""
    experiment_name: str
    is_multi_agent: bool
    training_episode_budget: int
    agent: AgentConfig
    env: EnvConfig

    @classmethod
    def from_json(cls, file_path: str) -> 'ExperimentConfig':
        """
        Loads an experiment configuration from a JSON file.
        This allows for the 'Configuration-Driven Development' pattern.
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        return cls(
            experiment_name=data.get("experiment_name", "Unnamed_Experiment"),
            is_multi_agent=data.get("is_multi_agent", False),
            training_episode_budget=data["training_episode_budget"],
            agent=AgentConfig(**data["agent"]),
            env=EnvConfig(**data["env"])
        )

    def save(self, file_path: str):
        """Saves current configuration to a JSON for reproducibility."""
        # Implementation for saving back to disk if needed
        pass