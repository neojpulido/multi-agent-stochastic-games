from typing import Dict, Any, Tuple

class StateHandler:
    """
    Transforms raw environment observations into compact, hashable 
    state representations for Tabular Q-Learning.
    """

    @staticmethod
    def get_agent_state(agent_id: str, observation: Dict[str, Any]) -> Tuple:
        """
        Extracts the partial observation for a specific agent.
        
        According to Stage 2 Specs:
        - Own location (x, y)
        - Whether it carries a sample (bool)
        - Binary state of the lake (bool)
        - (Does NOT see other agents' locations)
        
        Returns:
            Tuple: (pos_x, pos_y, has_sample, lake_flooded)
        """
        pos = observation["positions"][agent_id]
        has_sample = observation["has_sample"][agent_id]
        lake_flooded = observation["lake_flooded"]
        
        return (pos[0], pos[1], has_sample, lake_flooded)
