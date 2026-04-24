from src.core.actions import Actions, Directions

def test_map_coordinates():
    """
    Validates that (y, x) indexing matches the assignment's planet map.
    Origin (0,0) is Top-Left.
    """
    # 5x5 Grid center is (2,2)
    center = (2, 2)
    
    # North should be (1, 2) -> dy=-1
    assert Actions.apply_action(center, Directions.NORTH) == (1, 2)
    
    # South should be (3, 2) -> dy=+1
    assert Actions.apply_action(center, Directions.SOUTH) == (3, 2)
    
    # East should be (2, 3) -> dx=+1
    assert Actions.apply_action(center, Directions.EAST) == (2, 3)
    
    # West should be (2, 1) -> dx=-1
    assert Actions.apply_action(center, Directions.WEST) == (2, 1)
