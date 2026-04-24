from src.domain.gridworld.stochastic import StochasticMultiAgentEnv
from src.core.actions import Directions

def test_grid_initialization(default_env_config):
    env = StochasticMultiAgentEnv(config=default_env_config)
    assert env.config.grid_rows == 5
    assert env.config.grid_cols == 5
    assert env.locs["Lake"] == (2, 2)

def test_grid_reset(default_env_config):
    env = StochasticMultiAgentEnv(config=default_env_config)
    obs = env.reset()
    assert "positions" in obs
    assert obs["positions"]["Agent_A"] == (2, 0) # X
    assert obs["positions"]["Agent_B"] == (0, 2) # Y

def test_movement_and_pickup(default_env_config):
    env = StochasticMultiAgentEnv(config=default_env_config)
    env.reset()
    
    # Agent_A is at (2, 0). Move East to (2, 1)
    joint_action = {"Agent_A": Directions.EAST, "Agent_B": Directions.WAIT}
    obs, rewards, dones, _ = env.step(joint_action)
    
    assert obs["positions"]["Agent_A"] == (2, 1)
    assert rewards["Agent_A"] == -5.0 # Step cost
    assert rewards["Agent_B"] == -3.0 # Wait cost (-5 * 0.6)
