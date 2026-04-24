from qlearning import TransportGridWorld

def test_grid_initialization(default_env_config):
    env = TransportGridWorld(config=default_env_config)
    assert env.config.grid_rows == 3
    assert env.config.grid_cols == 3
    assert env.goal_coordinates == (2, 2)
    assert env.has_payload is False

def test_grid_reset(default_env_config):
    env = TransportGridWorld(config=default_env_config)
    state = env.reset()
    assert len(state) == 3
    assert env.agent_coordinates is not None
    assert env.pickup_coordinates != env.goal_coordinates

def test_automatic_pickup(default_env_config):
    env = TransportGridWorld(config=default_env_config)

    # Place agent adjacent to pickup (at 0,0 and pickup at 1,0)
    env.agent_coordinates = (0, 0)
    env.pickup_coordinates = (1, 0)
    env.has_payload = False

    # Move Right (index 1 is EAST) onto the pickup
    env.execute_step(1) 

    assert env.agent_coordinates == (1, 0)
    assert env.has_payload is True


