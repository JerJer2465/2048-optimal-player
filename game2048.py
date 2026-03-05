"""
2048 Game Implementation
Core game logic for the 2048 puzzle game.
"""

import random
import numpy as np
from typing import Tuple, List, Optional


class Game2048:
    """2048 game logic implementation."""
    
    def __init__(self, size: int = 4):
        """Initialize a new game.
        
        Args:
            size: Board size (default 4x4)
        """
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.score = 0
        self.game_over = False
        
        # Spawn two initial tiles
        self.spawn_tile()
        self.spawn_tile()
    
    def spawn_tile(self) -> bool:
        """Spawn a new tile (2 or 4) in a random empty cell.
        
        Returns:
            True if tile was spawned, False if board is full
        """
        empty_cells = list(zip(*np.where(self.board == 0)))
        if not empty_cells:
            return False
        
        row, col = random.choice(empty_cells)
        # 90% chance of 2, 10% chance of 4
        self.board[row, col] = 2 if random.random() < 0.9 else 4
        return True
    
    def move(self, direction: str) -> bool:
        """Execute a move in the specified direction.
        
        Args:
            direction: One of 'up', 'down', 'left', 'right'
            
        Returns:
            True if the move changed the board, False otherwise
        """
        if self.game_over:
            return False
        
        original_board = self.board.copy()
        
        if direction == 'left':
            self._move_left()
        elif direction == 'right':
            self._move_right()
        elif direction == 'up':
            self._move_up()
        elif direction == 'down':
            self._move_down()
        else:
            raise ValueError(f"Invalid direction: {direction}")
        
        # Check if board changed
        moved = not np.array_equal(original_board, self.board)
        
        if moved:
            self.spawn_tile()
            if not self.has_valid_moves():
                self.game_over = True
        
        return moved
    
    def _move_left(self):
        """Move and merge tiles to the left."""
        for i in range(self.size):
            self.board[i] = self._merge_row(self.board[i])
    
    def _move_right(self):
        """Move and merge tiles to the right."""
        for i in range(self.size):
            self.board[i] = self._merge_row(self.board[i][::-1])[::-1]
    
    def _move_up(self):
        """Move and merge tiles upward."""
        self.board = self.board.T
        self._move_left()
        self.board = self.board.T
    
    def _move_down(self):
        """Move and merge tiles downward."""
        self.board = self.board.T
        self._move_right()
        self.board = self.board.T
    
    def _merge_row(self, row: np.ndarray) -> np.ndarray:
        """Merge a single row to the left.
        
        Args:
            row: 1D array representing a row
            
        Returns:
            Merged row
        """
        # Remove zeros
        non_zero = row[row != 0]
        
        # Merge adjacent equal tiles
        merged = []
        skip = False
        for i in range(len(non_zero)):
            if skip:
                skip = False
                continue
            
            if i + 1 < len(non_zero) and non_zero[i] == non_zero[i + 1]:
                merged_value = non_zero[i] * 2
                merged.append(merged_value)
                self.score += merged_value
                skip = True
            else:
                merged.append(non_zero[i])
        
        # Pad with zeros
        merged.extend([0] * (self.size - len(merged)))
        return np.array(merged, dtype=int)
    
    def has_valid_moves(self) -> bool:
        """Check if any valid moves remain.
        
        Returns:
            True if moves are available, False otherwise
        """
        # Check for empty cells
        if np.any(self.board == 0):
            return True
        
        # Check for adjacent equal tiles
        for i in range(self.size):
            for j in range(self.size):
                current = self.board[i, j]
                # Check right neighbor
                if j < self.size - 1 and self.board[i, j + 1] == current:
                    return True
                # Check bottom neighbor
                if i < self.size - 1 and self.board[i + 1, j] == current:
                    return True
        
        return False
    
    def get_max_tile(self) -> int:
        """Get the maximum tile value on the board.
        
        Returns:
            Maximum tile value
        """
        return int(np.max(self.board))
    
    def get_state(self) -> Tuple[np.ndarray, int, bool]:
        """Get current game state.
        
        Returns:
            Tuple of (board, score, game_over)
        """
        return self.board.copy(), self.score, self.game_over
    
    def clone(self) -> 'Game2048':
        """Create a deep copy of the game state.
        
        Returns:
            New Game2048 instance with copied state
        """
        new_game = Game2048(self.size)
        new_game.board = self.board.copy()
        new_game.score = self.score
        new_game.game_over = self.game_over
        return new_game
    
    def __str__(self) -> str:
        """String representation of the board."""
        lines = [f"Score: {self.score}"]
        lines.append("-" * (self.size * 6 + 1))
        for row in self.board:
            lines.append("|" + "|".join(f"{val:5d}" if val else "     " for val in row) + "|")
            lines.append("-" * (self.size * 6 + 1))
        if self.game_over:
            lines.append("GAME OVER!")
        return "\n".join(lines)


def play_random_game():
    """Play a single game with random moves (demo)."""
    game = Game2048()
    moves = ['up', 'down', 'left', 'right']
    
    print("Starting game:")
    print(game)
    print()
    
    move_count = 0
    while not game.game_over:
        direction = random.choice(moves)
        if game.move(direction):
            move_count += 1
            if move_count % 10 == 0:
                print(f"Move {move_count}:")
                print(game)
                print()
    
    print("Final state:")
    print(game)
    print(f"\nTotal moves: {move_count}")
    print(f"Max tile: {game.get_max_tile()}")


if __name__ == "__main__":
    play_random_game()
