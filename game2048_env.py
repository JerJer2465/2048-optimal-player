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
MAX_EPISODE_STEPS = 3000   # hard cap — prevents infinite episodes


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
    Reward            :
        - Invalid move (board unchanged): -1
        - Valid move:  log2(score_delta + 1)
                     + 100 * log2(new_max_tile) when a new max tile is reached
    """

    metadata = {'render_modes': []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=OBS_SHAPE, dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.game: Game2048 = None
        self.max_tile_seen: int = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = Game2048()
        self.max_tile_seen = self.game.get_max_tile()
        self._steps = 0
        return encode_board(self.game.board), {}

    @staticmethod
    def _board_heuristic(board: np.ndarray) -> float:
        """Lightweight per-step shaping: empty cells + corner monotonicity.

        Kept small (scaled to ~0-1 range) so it doesn't overwhelm the merge
        score signal, but enough to guide the policy toward keeping big tiles
        in a corner and maintaining an organised board.
        """
        empty = float(np.sum(board == 0)) / 16.0          # 0-1

        # Monotonicity along rows and cols (reward sorted arrangements)
        mono = 0.0
        for row in board:
            nz = row[row > 0]
            if len(nz) > 1:
                if np.all(nz[:-1] >= nz[1:]) or np.all(nz[:-1] <= nz[1:]):
                    mono += 1.0
        for col in board.T:
            nz = col[col > 0]
            if len(nz) > 1:
                if np.all(nz[:-1] >= nz[1:]) or np.all(nz[:-1] <= nz[1:]):
                    mono += 1.0
        mono /= 8.0   # 0-1

        # Max tile in any corner bonus
        corners = [board[0,0], board[0,3], board[3,0], board[3,3]]
        max_tile = board.max()
        corner_bonus = 0.5 if max_tile > 0 and max_tile in corners else 0.0

        return 0.4 * empty + 0.4 * mono + 0.2 * corner_bonus

    def step(self, action: int):
        direction = DIRECTIONS[int(action)]
        prev_score = self.game.score
        prev_heuristic = self._board_heuristic(self.game.board)

        moved = self.game.move(direction)
        score_delta = self.game.score - prev_score

        if not moved:
            # Penalise invalid move; do NOT secretly apply another move —
            # that breaks the action→consequence relationship for learning.
            reward = -1.0
        else:
            # Merge reward (log-scaled to avoid early large merges dominating)
            reward = float(np.log2(score_delta + 1)) if score_delta > 0 else 0.0

            # One-time bonus each time a new max tile is reached
            new_max = self.game.get_max_tile()
            if new_max > self.max_tile_seen:
                reward += 10.0 * float(np.log2(max(new_max, 2)))
                self.max_tile_seen = new_max

            # Per-step board-structure shaping (small, ~0-1 scale)
            curr_heuristic = self._board_heuristic(self.game.board)
            reward += 0.5 * (curr_heuristic - prev_heuristic)

        self._steps += 1
        terminated = self.game.game_over or self._steps >= MAX_EPISODE_STEPS
        obs = encode_board(self.game.board)
        info = {
            'score': self.game.score,
            'max_tile': self.game.get_max_tile(),
            'moved': moved,
        }
        return obs, reward, terminated, False, info
