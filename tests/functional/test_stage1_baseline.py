from src.domain.gridworld.stochastic import StochasticMultiAgentEnv
from src.core.config import EnvConfig
from src.core.actions import Directions

def test_stage1_baseline_mechanics():
    """
    Ensures single-agent pickup/delivery logic works in the new MARL environment.
    """
    config = EnvConfig(
        grid_rows=5, grid_cols=5, p_flood=0.0, 
        step_cost=-5.0, success_reward=50.0, 
        collision_penalty=0.0, hazard_penalty=0.0
    )
    env = StochasticMultiAgentEnv(config)
    env.reset()
    
    # Force Agent_A state: at U (2,4) with NO sample
    env.positions["Agent_A"] = (2, 4)
    env.has_sample["Agent_A"] = False
    
    # Step 1: Automatic Pickup at U
    obs, rewards, dones, _ = env.step({"Agent_A": Directions.WAIT, "Agent_B": Directions.WAIT})
    assert obs["has_sample"]["Agent_A"] is True
    assert rewards["Agent_A"] == 10.0 - 3.0 # Pickup Reward + Wait Cost
    
    # Step 2: Delivery at X (2,0)
    env.positions["Agent_A"] = (2, 0)
    obs, rewards, dones, _ = env.step({"Agent_A": Directions.WAIT, "Agent_B": Directions.WAIT})
    assert dones["Agent_A"] is True
    assert rewards["Agent_A"] == 50.0 - 3.0 # Success Reward + Wait Cost
