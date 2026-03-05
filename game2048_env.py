"""
Gymnasium environment wrapper for the 2048 game.
Used for reinforcement learning training via PPO.
"""

import sys
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game2048 import Game2048

DIRECTIONS = ['up', 'down', 'left', 'right']
N_ACTIONS = 4
OBS_SHAPE = (16, 4, 4)
MAX_EPISODE_STEPS = 3000   # safety cap — should rarely trigger with action masking


def encode_board(board: np.ndarray) -> np.ndarray:
    """One-hot encode a 2048 board into a (16, 4, 4) float32 array.

    Channel 0  = empty cell (value 0).
    Channel k  = tile with value 2^k, for k in 1..15.
    """
    encoded = np.zeros(OBS_SHAPE, dtype=np.float32)
    for i in range(4):
        for j in range(4):
            val = int(board[i, j])
            if val == 0:
                encoded[0, i, j] = 1.0
            else:
                channel = int(np.log2(val))
                encoded[min(channel, 15), i, j] = 1.0
    return encoded


class Game2048Env(gym.Env):
    """Gymnasium environment for the 2048 game.

    Observation space : Box(16, 4, 4) one-hot encoded board.
    Action space      : Discrete(4)  — 0=up, 1=down, 2=left, 3=right.
    Reward            : log2(score_delta + 1) for any merge, 0 otherwise.

    Use action_mask() before sampling to ensure only valid actions are chosen.
    Invalid actions passed to step() still execute (with 0 reward if no merge)
    but the caller is responsible for masking them out.
    """

    metadata = {'render_modes': []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=OBS_SHAPE, dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.game: Game2048 = None
        self._steps: int = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = Game2048()
        self._steps = 0
        return encode_board(self.game.board), {}

    def action_mask(self) -> np.ndarray:
        """Return bool[4] — True if direction i produces a valid (changed) board."""
        mask = np.zeros(4, dtype=bool)
        for i, direction in enumerate(DIRECTIONS):
            test = self.game.clone()
            mask[i] = test.move(direction)
        return mask

    def step(self, action: int):
        direction = DIRECTIONS[int(action)]
        prev_score = self.game.score

        self.game.move(direction)
        score_delta = self.game.score - prev_score

        reward = float(np.log2(score_delta + 1)) if score_delta > 0 else 0.0

        self._steps += 1
        terminated = self.game.game_over or self._steps >= MAX_EPISODE_STEPS
        obs = encode_board(self.game.board)
        info = {
            'score': self.game.score,
            'max_tile': self.game.get_max_tile(),
        }
        return obs, reward, terminated, False, info
