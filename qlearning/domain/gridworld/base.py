from qlearning.core.environment import BaseEnvironment
from qlearning.core.actions import Actions

class GridWorldBase(BaseEnvironment):
    """Core physics shared by both Stage 1 and Stage 2."""
    def __init__(self, config):
        self.cfg = config
        self.rows = config.grid_rows
        self.cols = config.grid_cols
        self.goal_pos = (self.rows - 1, self.cols - 1)

    def _is_within_bounds(self, pos):
        return 0 <= pos[0] < self.rows and 0 <= pos[1] < self.cols