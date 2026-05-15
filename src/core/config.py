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
    action_size: int


@dataclass(frozen=True)
class EnvConfig:
    """Structural parameters and reward values for the Rugged Planet MARL GridWorld."""
    grid_rows: int
    grid_cols: int
    p_flood: float
    step_cost: float
    wait_cost: float       # required — was missing in original
    pickup_reward: float   # required — was missing in original
    success_reward: float
    collision_penalty: float = 0.0
    hazard_penalty: float = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    """Root configuration object for a simulation run."""
    experiment_name: str
    is_multi_agent: bool
    training_episode_budget: int
    agent: AgentConfig
    env: EnvConfig

    @classmethod
    def from_json(cls, file_path: str) -> 'ExperimentConfig':
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(
            experiment_name=data.get("experiment_name", "Unnamed_Experiment"),
            is_multi_agent=data.get("is_multi_agent", False),
            training_episode_budget=data["training_episode_budget"],
            agent=AgentConfig(**data["agent"]),
            env=EnvConfig(**data["env"])
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        return cls(
            experiment_name=data.get("experiment_name", "Unnamed_Experiment"),
            is_multi_agent=data.get("is_multi_agent", True),
            training_episode_budget=data["training_episode_budget"],
            agent=AgentConfig(**data["agent"]),
            env=EnvConfig(**data["env"])
        )
