"""
AI Player for 2048 using Expectimax algorithm.
Implements optimal play strategy to maximize score.
"""

import numpy as np
from typing import Tuple, Optional
from game2048 import Game2048


class AIPlayer:
    """AI player using Expectimax algorithm with heuristics."""
    
    def __init__(self, search_depth: int = 3):
        """Initialize AI player.
        
        Args:
            search_depth: How many moves to look ahead (default 3)
        """
        self.search_depth = search_depth
        self.moves_tried = 0
        
    def get_best_move(self, game: Game2048) -> Optional[str]:
        """Find the best move for current game state.
        
        Args:
            game: Current game state
            
        Returns:
            Best move ('up', 'down', 'left', 'right') or None
        """
        self.moves_tried = 0
        best_move = None
        best_score = -float('inf')
        
        for direction in ['up', 'down', 'left', 'right']:
            # Try this move
            test_game = game.clone()
            if test_game.move(direction):
                # Evaluate using expectimax
                score = self._expectimax(test_game, self.search_depth - 1, False)
                
                if score > best_score:
                    best_score = score
                    best_move = direction
        
        return best_move
    
    def _expectimax(self, game: Game2048, depth: int, is_player_turn: bool) -> float:
        """Expectimax algorithm implementation.
        
        Args:
            game: Current game state
            depth: Remaining search depth
            is_player_turn: True if player's turn, False if chance (tile spawn)
            
        Returns:
            Expected score for this state
        """
        self.moves_tried += 1
        
        # Terminal conditions
        if depth == 0 or game.game_over:
            return self._evaluate_board(game)
        
        if is_player_turn:
            # Maximize: try all moves and pick best
            max_score = -float('inf')
            for direction in ['up', 'down', 'left', 'right']:
                test_game = game.clone()
                if test_game.move(direction):
                    score = self._expectimax(test_game, depth - 1, False)
                    max_score = max(max_score, score)
            
            return max_score if max_score != -float('inf') else 0
        else:
            # Chance node: average over possible tile spawns
            empty_cells = list(zip(*np.where(game.board == 0)))
            if not empty_cells:
                return self._evaluate_board(game)
            
            expected_score = 0.0
            num_empty = len(empty_cells)
            
            # Sample a subset of empty cells to reduce branching
            sample_size = min(num_empty, 4)  # Only check up to 4 cells
            sampled_cells = empty_cells[:sample_size]
            
            for row, col in sampled_cells:
                # 90% chance of spawning 2
                test_game_2 = game.clone()
                test_game_2.board[row, col] = 2
                score_2 = self._expectimax(test_game_2, depth - 1, True)
                expected_score += 0.9 * score_2
                
                # 10% chance of spawning 4
                test_game_4 = game.clone()
                test_game_4.board[row, col] = 4
                score_4 = self._expectimax(test_game_4, depth - 1, True)
                expected_score += 0.1 * score_4
            
            return expected_score / sample_size
    
    def _evaluate_board(self, game: Game2048) -> float:
        """Evaluate board state using heuristics.
        
        Combines multiple heuristics:
        - Monotonicity: tiles should increase in one direction
        - Smoothness: adjacent tiles should be similar
        - Empty cells: more empty = better
        - Max tile position: keep max in corner
        
        Args:
            game: Game state to evaluate
            
        Returns:
            Heuristic score (higher is better)
        """
        board = game.board
        score = 0.0
        
        # Weight factors (tuned for good performance)
        MONOTONICITY_WEIGHT = 1.0
        SMOOTHNESS_WEIGHT = 0.1
        EMPTY_WEIGHT = 2.7
        MAX_CORNER_WEIGHT = 1.0
        
        # 1. Monotonicity: reward increasing/decreasing sequences
        score += MONOTONICITY_WEIGHT * self._monotonicity(board)
        
        # 2. Smoothness: penalize large differences between adjacent tiles
        score -= SMOOTHNESS_WEIGHT * self._smoothness(board)
        
        # 3. Empty cells: reward having more empty spaces
        empty_count = np.count_nonzero(board == 0)
        score += EMPTY_WEIGHT * empty_count
        
        # 4. Max tile in corner: reward keeping max tile in corner
        max_val = np.max(board)
        corners = [board[0, 0], board[0, -1], board[-1, 0], board[-1, -1]]
        if max_val in corners:
            score += MAX_CORNER_WEIGHT * max_val
        
        # Add game score as baseline
        score += game.score
        
        return score
    
    def _monotonicity(self, board: np.ndarray) -> float:
        """Calculate monotonicity score.
        
        Measures how well tiles form increasing/decreasing sequences
        in rows and columns.
        """
        totals = [0, 0, 0, 0]  # up, down, left, right
        
        # Check rows (left-right)
        for row in board:
            current = 0
            next_idx = current + 1
            while next_idx < len(row):
                while next_idx < len(row) and row[next_idx] == 0:
                    next_idx += 1
                if next_idx >= len(row):
                    next_idx -= 1
                
                current_val = np.log2(row[current]) if row[current] > 0 else 0
                next_val = np.log2(row[next_idx]) if row[next_idx] > 0 else 0
                
                if current_val > next_val:
                    totals[2] += next_val - current_val  # left
                elif next_val > current_val:
                    totals[3] += current_val - next_val  # right
                
                current = next_idx
                next_idx += 1
        
        # Check columns (up-down)
        for col in board.T:
            current = 0
            next_idx = current + 1
            while next_idx < len(col):
                while next_idx < len(col) and col[next_idx] == 0:
                    next_idx += 1
                if next_idx >= len(col):
                    next_idx -= 1
                
                current_val = np.log2(col[current]) if col[current] > 0 else 0
                next_val = np.log2(col[next_idx]) if col[next_idx] > 0 else 0
                
                if current_val > next_val:
                    totals[0] += next_val - current_val  # up
                elif next_val > current_val:
                    totals[1] += current_val - next_val  # down
                
                current = next_idx
                next_idx += 1
        
        return max(totals[0], totals[1]) + max(totals[2], totals[3])
    
    def _smoothness(self, board: np.ndarray) -> float:
        """Calculate smoothness penalty.
        
        Measures differences between adjacent tiles.
        Lower smoothness = larger differences = worse.
        """
        smoothness = 0
        
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if board[i, j] != 0:
                    value = np.log2(board[i, j])
                    
                    # Check right neighbor
                    if j + 1 < board.shape[1] and board[i, j + 1] != 0:
                        target_value = np.log2(board[i, j + 1])
                        smoothness += abs(value - target_value)
                    
                    # Check down neighbor
                    if i + 1 < board.shape[0] and board[i + 1, j] != 0:
                        target_value = np.log2(board[i + 1, j])
                        smoothness += abs(value - target_value)
        
        return smoothness


def play_game_with_ai(search_depth: int = 3, verbose: bool = True) -> Tuple[int, int, int]:
    """Play a full game using the AI player.
    
    Args:
        search_depth: Expectimax search depth
        verbose: Print progress during game
        
    Returns:
        Tuple of (final_score, max_tile, moves_made)
    """
    game = Game2048()
    ai = AIPlayer(search_depth=search_depth)
    moves_made = 0
    
    if verbose:
        print(f"Starting game with AI (search depth: {search_depth})")
        print(game)
        print()
    
    while not game.game_over:
        move = ai.get_best_move(game)
        
        if move is None:
            break
        
        if game.move(move):
            moves_made += 1
            
            if verbose and moves_made % 50 == 0:
                print(f"Move {moves_made}: {move}")
                print(game)
                print()
    
    if verbose:
        print("Game Over!")
        print(game)
        print(f"\nFinal score: {game.score}")
        print(f"Max tile: {game.get_max_tile()}")
        print(f"Total moves: {moves_made}")
    
    return game.score, game.get_max_tile(), moves_made


if __name__ == "__main__":
    # Run a single game
    score, max_tile, moves = play_game_with_ai(search_depth=3, verbose=True)
