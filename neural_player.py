"""
Neural network model and inference player for 2048.

TwoFortyEightNet: shared-trunk CNN with a policy head and a value head.
NeuralPlayer:     loads a trained checkpoint and provides get_best_move()
                  with the same interface as AIPlayer.
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


class TwoFortyEightNet(nn.Module):
    """Shared-trunk CNN for 2048 policy and value estimation.

    Input : (B, 16, 4, 4) one-hot encoded board.
    Output: policy logits (B, 4),  value estimate (B, 1).
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(16, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.policy_head = nn.Linear(512, 4)
        self.value_head = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.policy_head(x), self.value_head(x)


class NeuralPlayer:
    """Inference wrapper that loads a trained checkpoint and picks moves.

    Selects the highest-probability valid move according to the policy
    head.  Falls back gracefully when no checkpoint exists.
    """

    def __init__(self, checkpoint_path: str = CHECKPOINT_PATH):
        self.device = torch.device('cpu')
        self.model = TwoFortyEightNet().to(self.device)
        self.checkpoint_path = checkpoint_path
        self.checkpoint_mtime: Optional[float] = None
        self.loaded = False
        self.episodes_done: int = 0
        self.avg_score: float = 0.0
        self.best_score: int = 0
        self._try_load()

    # ── Loading ────────────────────────────────────────────────────────────

    def _try_load(self) -> bool:
        if not os.path.exists(self.checkpoint_path):
            return False
        try:
            mtime = os.path.getmtime(self.checkpoint_path)
            ckpt = torch.load(
                self.checkpoint_path, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.model.eval()
            self.checkpoint_mtime = mtime
            self.loaded = True
            self.episodes_done = int(ckpt.get('episodes_done', 0))
            self.avg_score = float(ckpt.get('avg_score_recent', 0.0))
            self.best_score = int(ckpt.get('best_score', 0))
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
        """Return the best valid move direction for the current board."""
        from game2048_env import encode_board

        obs = torch.tensor(
            encode_board(game.board), dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            logits, _ = self.model(obs)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        # Try directions in descending probability order, skip invalid moves
        for action in np.argsort(probs)[::-1]:
            direction = DIRECTIONS[int(action)]
            test_game = game.clone()
            if test_game.move(direction):
                return direction

        return None

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def is_available(checkpoint_path: str = CHECKPOINT_PATH) -> bool:
        return os.path.exists(checkpoint_path)

    def status_line(self) -> str:
        """Short human-readable status string for the UI header."""
        if not self.loaded:
            return "Neural AI (no checkpoint)"
        return (
            f"Neural AI  ep:{self.episodes_done:,}  "
            f"avg:{self.avg_score:.0f}  best:{self.best_score:,}"
        )
