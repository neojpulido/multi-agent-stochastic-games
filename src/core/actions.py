from enum import IntEnum
from typing import Tuple, List, Dict

class Directions(IntEnum):
    """
    Semantic enumeration of cardinal and ordinal directions.
    Compatible with Stage 1 and Stage 2.
    """
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3
    WAIT = 4
    NORTH_EAST = 5
    NORTH_WEST = 6
    SOUTH_EAST = 7
    SOUTH_WEST = 8

class Actions:
    """
    Handles coordinate geometry using (y, x) indexing.
    - (0, 2) is North (Y)
    - (4, 2) is South (V)
    - (2, 0) is West (X)
    - (2, 4) is East (U)
    """
    
    # Map Directions to (dy, dx)
    _DELTAS: Dict[Directions, Tuple[int, int]] = {
        Directions.NORTH: (-1, 0),
        Directions.SOUTH: (1, 0),
        Directions.EAST: (0, 1),
        Directions.WEST: (0, -1),
        Directions.WAIT: (0, 0),
        Directions.NORTH_EAST: (-1, 1),
        Directions.NORTH_WEST: (-1, -1),
        Directions.SOUTH_EAST: (1, 1),
        Directions.SOUTH_WEST: (1, -1),
    }

    @staticmethod
    def get_delta(action: int) -> Tuple[int, int]:
        direction = Directions(action)
        return Actions._DELTAS.get(direction, (0, 0))

    @staticmethod
    def apply_action(current_pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        dy, dx = Actions.get_delta(action)
        return (current_pos[0] + dy, current_pos[1] + dx)

    @staticmethod
    def is_valid_move(pos: Tuple[int, int], grid_size: Tuple[int, int]) -> bool:
        y, x = pos
        rows, cols = grid_size
        return 0 <= y < rows and 0 <= x < cols

    @staticmethod
    def get_action_space(num_actions: int) -> List[int]:
        return list(range(num_actions))
