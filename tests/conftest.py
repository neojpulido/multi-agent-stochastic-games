import pytest
from src.core.config import AgentConfig, EnvConfig, ExperimentConfig

@pytest.fixture
def default_agent_config():
    return AgentConfig(
        learning_rate_alpha=0.1,
        discount_factor_gamma=0.9,
        initial_epsilon=1.0,
        epsilon_decay_rate=0.99,
        minimum_epsilon=0.01,
        action_size=5
    )

@pytest.fixture
def default_env_config():
    return EnvConfig(
        grid_rows=5,
        grid_cols=5,
        p_flood=0.1,
        step_cost=-5.0,
        success_reward=50.0,
        collision_penalty=-20.0,
        hazard_penalty=-20.0
    )

@pytest.fixture
def default_experiment_config(default_agent_config, default_env_config):
    return ExperimentConfig(
        experiment_name="Test_Experiment",
        is_multi_agent=True,
        training_episode_budget=2000,
        agent=default_agent_config,
        env=default_env_config
    )
