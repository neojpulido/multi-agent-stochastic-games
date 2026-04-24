from src.domain.gridworld.stochastic import StochasticMultiAgentEnv
from src.core.config import EnvConfig
from src.core.actions import Directions

def test_stage2_marl_dynamics():
    """
    Validates Stage 2 specific dynamics: Collisions, Water Hazard, and Stochasticity.
    """
    config = EnvConfig(
        grid_rows=5, grid_cols=5, p_flood=0.0, # Disable auto-toggling for controlled test
        step_cost=-5.0, success_reward=50.0, 
        collision_penalty=-20.0, hazard_penalty=-20.0
    )
    env = StochasticMultiAgentEnv(config)
    env.reset()
    
    # Test 1: Simultaneous Collision in the Lake (2,2)
    env.positions["Agent_A"] = (2, 2)
    env.positions["Agent_B"] = (2, 2)
    
    # Perform a step
    _, rewards, _, _ = env.step({"Agent_A": Directions.WAIT, "Agent_B": Directions.WAIT})
    
    # Both should get collision penalty (-20) + Wait cost (-3) = -23
    assert rewards["Agent_A"] == -23.0
    assert rewards["Agent_B"] == -23.0
    
    # Test 2: Water Hazard (Sarah's Phase 1)
    env.reset()
    env.lake_flooded = True 
    env.positions["Agent_A"] = (2, 2)
    env.positions["Agent_B"] = (0, 0) # Move B away
    
    _, rewards, _, _ = env.step({"Agent_A": Directions.WAIT, "Agent_B": Directions.WAIT})
    
    # Agent A: Hazard (-20) + Wait (-3) = -23
    # Agent B: Wait (-3) = -3
    assert rewards["Agent_A"] == -23.0
    assert rewards["Agent_B"] == -3.0
