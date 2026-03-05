"""
Neural network model and inference player for 2048.

TwoFortyEightNet        : shared-trunk CNN — policy head + value head.
NeuralExpectimaxPlayer  : runs depth-3 Expectimax using the NN *value head*
                          at leaf nodes (AlphaZero-style). Used during play
                          and self-play data generation.
NeuralPlayer            : checkpoint manager that wraps NeuralExpectimaxPlayer
                          and exposes the same interface as AIPlayer.
"""

import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game2048 import Game2048

DIRECTIONS = ['up', 'down', 'left', 'right']
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'model_checkpoint.pth'
)

# ── Model ─────────────────────────────────────────────────────────────────────

class TwoFortyEightNet(nn.Module):
    """Shared-trunk CNN for 2048 policy and value estimation.

    Input : (B, 16, 4, 4) one-hot encoded board.
            Channel 0 = empty cell, channel k = tile 2^k (k = 1…15).
    Output: policy logits (B, 4),  value estimate (B, 1).
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(16, 128, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.fc1         = nn.Linear(128 * 4 * 4, 512)
        self.policy_head = nn.Linear(512, 4)
        self.value_head  = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.policy_head(x), self.value_head(x)


# ── Search player ──────────────────────────────────────────────────────────────

class NeuralExpectimaxPlayer:
    """AlphaZero-style player: Expectimax search with the NN value head at leaves.

    The hand-crafted _evaluate_board heuristic is replaced entirely by the
    learned value head. With depth=3 the search evaluates up to a few hundred
    positions per move and is fast enough for real-time play.

    Parameters
    ----------
    model       : TwoFortyEightNet (already on the correct device, eval mode).
    device      : torch device.
    search_depth: Expectimax depth. 3 is fast; 4 is stronger but slower.
    """

    # Maximum chance-node samples at each depth level (keep search tractable).
    _CHANCE_SAMPLES = {0: 2, 1: 3}   # depth ≤ 0 → 2 cells, depth 1 → 3 cells
    _CHANCE_SAMPLES_DEFAULT = 4       # depth ≥ 2

    def __init__(self, model: TwoFortyEightNet, device: torch.device,
                 search_depth: int = 3):
        self.model        = model
        self.device       = device
        self.search_depth = search_depth
        self._cache: dict = {}

    # ── Public interface ───────────────────────────────────────────────────────

    def get_best_move(self, game: Game2048) -> Optional[str]:
        """Return the highest-value valid move direction."""
        if len(self._cache) > 300_000:
            self._cache.clear()

        best_move  = None
        best_value = -float('inf')

        for direction in DIRECTIONS:
            child = game.clone()
            if child.move(direction):
                val = self._expectimax(child, self.search_depth - 1, is_player=False)
                if val > best_value:
                    best_value = val
                    best_move  = direction

        return best_move

    # ── Search internals ──────────────────────────────────────────────────────

    def _expectimax(self, game: Game2048, depth: int, is_player: bool) -> float:
        if depth == 0 or game.game_over:
            return self._nn_value(game)

        key = (game.board.tobytes(), depth, is_player)
        if key in self._cache:
            return self._cache[key]

        if is_player:
            result = self._max_node(game, depth)
        else:
            result = self._chance_node(game, depth)

        self._cache[key] = result
        return result

    def _max_node(self, game: Game2048, depth: int) -> float:
        best = -float('inf')
        any_valid = False
        for direction in DIRECTIONS:
            child = game.clone()
            if child.move(direction):
                any_valid = True
                val = self._expectimax(child, depth - 1, is_player=False)
                if val > best:
                    best = val
        return best if any_valid else self._nn_value(game)

    def _chance_node(self, game: Game2048, depth: int) -> float:
        empty = list(zip(*np.where(game.board == 0)))
        if not empty:
            return self._nn_value(game)

        n_samples = self._CHANCE_SAMPLES.get(depth, self._CHANCE_SAMPLES_DEFAULT)
        sampled   = empty[: min(len(empty), n_samples)]

        total = 0.0
        for row, col in sampled:
            for tile_val, prob in ((2, 0.9), (4, 0.1)):
                child = game.clone()
                child.board[row, col] = tile_val
                total += prob * self._expectimax(child, depth - 1, is_player=True)

        return total / len(sampled)

    def _nn_value(self, game: Game2048) -> float:
        """Call the NN value head on a single board."""
        from game2048_env import encode_board
        obs = torch.tensor(
            encode_board(game.board), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            _, value = self.model(obs)
        return float(value.item())


# ── Checkpoint manager ────────────────────────────────────────────────────────

class NeuralPlayer:
    """Loads a trained checkpoint and provides get_best_move() via Expectimax search.

    Drop-in replacement for AIPlayer with the same interface.
    """

    def __init__(self, checkpoint_path: str = CHECKPOINT_PATH,
                 search_depth: int = 3):
        self.device         = torch.device('cpu')
        self.model          = TwoFortyEightNet().to(self.device)
        self.checkpoint_path = checkpoint_path
        self.search_depth   = search_depth
        self.checkpoint_mtime: Optional[float] = None
        self.loaded         = False
        self.episodes_done  = 0
        self.avg_score      = 0.0
        self.best_score     = 0
        self._player: Optional[NeuralExpectimaxPlayer] = None
        self._try_load()

    # ── Loading ────────────────────────────────────────────────────────────

    def _try_load(self) -> bool:
        if not os.path.exists(self.checkpoint_path):
            return False
        try:
            mtime = os.path.getmtime(self.checkpoint_path)
            ckpt  = torch.load(
                self.checkpoint_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.model.eval()
            self.checkpoint_mtime = mtime
            self.loaded           = True
            self.episodes_done    = int(ckpt.get('episodes_done', 0))
            self.avg_score        = float(ckpt.get('avg_score_recent', 0.0))
            self.best_score       = int(ckpt.get('best_score', 0))
            self._player          = NeuralExpectimaxPlayer(
                self.model, self.device, self.search_depth
            )
            return True
        except Exception:
            return False

    def reload_if_updated(self) -> bool:
        """Reload from disk if the checkpoint file has been modified."""
        if not os.path.exists(self.checkpoint_path):
            return False
        try:
            mtime = os.path.getmtime(self.checkpoint_path)
        except OSError:
            return False
        if mtime != self.checkpoint_mtime:
            return self._try_load()
        return False

    # ── Inference ──────────────────────────────────────────────────────────

    def get_best_move(self, game: Game2048) -> Optional[str]:
        """Run Expectimax with NN value head. Falls back to highest-prob policy
        if no checkpoint is loaded."""
        if self._player is not None:
            return self._player.get_best_move(game)

        # Fallback: raw policy (no search), used before first checkpoint
        from game2048_env import encode_board
        obs = torch.tensor(
            encode_board(game.board), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.model(obs)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        for action in np.argsort(probs)[::-1]:
            direction = DIRECTIONS[int(action)]
            test = game.clone()
            if test.move(direction):
                return direction
        return None

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def is_available(checkpoint_path: str = CHECKPOINT_PATH) -> bool:
        return os.path.exists(checkpoint_path)

    def status_line(self) -> str:
        if not self.loaded:
            return f"Neural+Search d={self.search_depth} (no checkpoint)"
        return (
            f"Neural+Search d={self.search_depth}  "
            f"ep:{self.episodes_done:,}  "
            f"avg:{self.avg_score:.0f}  best:{self.best_score:,}"
        )
