from typing import Dict, Any, Tuple


class StateHandler:
    """
    Handles the transformation of raw environment observations into
    hashable state representations for Tabular Q-Learning.
    """

    @staticmethod
    def to_tabular_key(agent_id: str, observation: Dict[str, Any]) -> Tuple:
        """
        Flattens the observation dictionary into a unique tuple key.

        Example Output for Sarah:
        ((2, 3), (2, 4), True, True)
        -> (Self_Pos, Other_Pos, Lake_Flooded, Has_Item)
        """
        # 1. Get Agent's own position
        self_pos = observation["agent_positions"][agent_id]

        # 2. Get Other Agents' positions (sorted by ID to ensure consistency)
        # Sorting is a professional touch: it ensures 'Sarah+Robert' is the
        # same state as 'Robert+Sarah'.
        other_positions = tuple(
            pos for aid, pos in sorted(observation["agent_positions"].items())
            if aid != agent_id
        )

        # 3. Environmental Stochasticity
        lake_status = observation.get("lake_flooded", False)

        # 4. Task Progress
        # In Assignment 1 & 2, the agent needs to know if it has the item
        # because the 'goal' changes from Item A to Target B.
        has_item = observation.get("has_item", {}).get(agent_id, False)

        # Return a single hashable tuple
        return (self_pos, other_positions, lake_status, has_item)