from enum import IntEnum
from typing import Tuple, Optional

class Directions(IntEnum):
    """
    Semantic enumeration of the Action Space (A).
    Maps cardinal and ordinal directions to discrete action indices.
    """
    WEST = 0
    EAST = 1
    SOUTH = 2
    NORTH = 3
    NORTH_WEST = 4
    NORTH_EAST = 5
    SOUTH_WEST = 6
    SOUTH_EAST = 7
    WAIT = 8

    @property
    def coordinate_delta(self) -> Tuple[int, int]:
        """Returns the (dx, dy) coordinate change for this direction."""
        return Actions.map_direction_to_unit_vector(self)

class Actions:
    """
    Utility class for manipulating the Action Space and coordinate geometry.
    Maps the abstract action indices to physical grid transitions.
    """
    
    # Internal mapping for unit vector calculations
    _DIRECTION_TO_VECTOR_MAP = {
        Directions.WEST: (-1, 0),
        Directions.EAST: (1, 0),
        Directions.SOUTH: (0, -1),
        Directions.NORTH: (0, 1),
        Directions.NORTH_WEST: (-1, -1),
        Directions.NORTH_EAST: (-1, 1),
        Directions.SOUTH_WEST: (1, -1),
        Directions.SOUTH_EAST: (1, 1),
        Directions.WAIT: (0, 0)
    }

    @staticmethod
    def get_inverse_direction(direction: Directions) -> Directions:
        """Calculates the mathematically opposite direction (180-degree flip)."""
        if direction == Directions.WAIT:
            return Directions.WAIT

        inverse_mapping = {
            Directions.NORTH: Directions.SOUTH,
            Directions.SOUTH: Directions.NORTH,
            Directions.EAST: Directions.WEST,
            Directions.WEST: Directions.EAST,
            Directions.NORTH_WEST: Directions.SOUTH_EAST,
            Directions.SOUTH_EAST: Directions.NORTH_WEST,
            Directions.NORTH_EAST: Directions.SOUTH_WEST,
            Directions.SOUTH_WEST: Directions.NORTH_EAST,
        }
        return inverse_mapping[direction]

    @staticmethod
    def map_direction_to_unit_vector(direction: Directions) -> Tuple[int, int]:
        """Converts a Directions enum to a (dx, dy) unit vector tuple."""
        return Actions._DIRECTION_TO_VECTOR_MAP[direction]

    @staticmethod
    def map_unit_vector_to_direction(unit_vector: Tuple[int, int]) -> Optional[Directions]:
        """Maps a raw (dx, dy) vector back to its semantic Directions enum."""
        dx = 1 if unit_vector[0] > 0 else (-1 if unit_vector[0] < 0 else 0)
        dy = 1 if unit_vector[1] > 0 else (-1 if unit_vector[1] < 0 else 0)
        
        for direction, mapped_vector in Actions._DIRECTION_TO_VECTOR_MAP.items():
            if (dx, dy) == mapped_vector:
                return direction
        return None

    @staticmethod
    def calculate_next_coordinates(current_coordinates: Tuple[int, int], action_index: int) -> Tuple[int, int]:
        """Computes the successor state coordinates after applying an action."""
        dx, dy = Actions._DIRECTION_TO_VECTOR_MAP[Directions(action_index)]
        return (current_coordinates[0] + dx, current_coordinates[1] + dy)

    @staticmethod
    def is_within_boundaries(coordinates: Tuple[int, int], rows: int, cols: int) -> bool:
        """Validates if a coordinate pair exists within the defined grid boundaries."""
        x, y = coordinates
        return 0 <= x < rows and 0 <= y < cols
