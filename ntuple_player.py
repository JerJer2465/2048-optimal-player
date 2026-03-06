"""
N-Tuple network player for the 2048 UI.

Loads weights from ntuple.bin and exposes the same get_best_move(game)
interface as AIPlayer and NeuralPlayer.  Falls back gracefully when no
weights file exists yet.
"""
import os
import math

import numpy as np

from ntuple_network import board, pattern, learning

HERE         = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(HERE, 'ntuple.bin')

# Must match the patterns used in td_trainer.py
_PATTERNS = [
    [0, 1, 2, 3, 4, 5],
    [4, 5, 6, 7, 8, 9],
    [0, 1, 2, 4, 5, 6],
    [4, 5, 6, 8, 9, 10],
]

# board.move() opcodes → direction strings
_OPCODE_TO_DIR = ['up', 'right', 'down', 'left']


def _numpy_to_board(np_board: np.ndarray) -> board:
    """Convert a 4×4 numpy array of tile values to a bitboard.

    game2048.py stores actual tile values (2, 4, 8, …); the bitboard stores
    their log₂ (1, 2, 3, …), with 0 for empty cells.
    """
    b = board(0)
    for row in range(4):
        for col in range(4):
            val = int(np_board[row, col])
            b.set(row * 4 + col, int(math.log2(val)) if val > 0 else 0)
    return b


class NTuplePlayer:
    """
    2048 player backed by a trained N-Tuple TD network.

    Compatible with AIPlayer and NeuralPlayer — just call get_best_move(game).

    Attributes:
        loaded: True if weights were successfully loaded from disk.
    """

    def __init__(self, weights_path: str = WEIGHTS_PATH):
        board.lookup.init()

        self._tdl = learning()
        for patt in _PATTERNS:
            self._tdl.add_feature(pattern(patt))

        self._weights_path = weights_path
        self.loaded = self._tdl.load(weights_path)

    @classmethod
    def is_available(cls, weights_path: str = WEIGHTS_PATH) -> bool:
        """Return True if a weights file exists at weights_path."""
        return os.path.exists(weights_path)

    def get_best_move(self, game) -> str | None:
        """
        Return the best direction for the current Game2048 state.

        Returns 'up', 'right', 'down', or 'left', or None if the game is over.
        """
        if not self.loaded:
            return None
        b  = _numpy_to_board(game.board)
        mv = self._tdl.select_best_move(b)
        return _OPCODE_TO_DIR[mv.action()] if mv.is_valid() else None
