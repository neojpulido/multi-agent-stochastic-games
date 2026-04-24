import pytest
from qlearning import AgentConfig, EnvConfig, ExperimentConfig

@pytest.fixture
def default_agent_config():
    return AgentConfig()

@pytest.fixture
def default_env_config():
    return EnvConfig(grid_rows=3, grid_cols=3) # Small grid for unit tests

@pytest.fixture
def default_experiment_config(default_agent_config, default_env_config):
    return ExperimentConfig(agent=default_agent_config, env=default_env_config)
