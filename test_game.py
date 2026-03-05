"""
Test script for 2048 game logic.
Verifies core functionality works correctly.
"""

from game2048 import Game2048
import numpy as np


def test_initialization():
    """Test game initialization."""
    game = Game2048()
    assert game.size == 4
    assert game.score == 0
    assert not game.game_over
    # Should have exactly 2 tiles at start
    assert np.count_nonzero(game.board) == 2
    print("✓ Initialization test passed")


def test_move_mechanics():
    """Test basic move mechanics."""
    game = Game2048()
    
    # Set up a known board state
    game.board = np.array([
        [2, 2, 0, 0],
        [4, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    
    # Test left move (should merge the 2s)
    game.move('left')
    assert game.board[0, 0] == 4  # 2+2=4
    print("✓ Move mechanics test passed")


def test_merge_scoring():
    """Test that merges update score correctly."""
    game = Game2048()
    game.board = np.array([
        [2, 2, 4, 4],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    
    initial_score = game.score
    game.move('left')
    # Should merge 2+2=4 (score +4) and 4+4=8 (score +8)
    # New score = initial + 4 + 8 = initial + 12
    # But spawning a new tile happens, so we just check it increased
    assert game.score >= initial_score + 12
    print("✓ Merge scoring test passed")


def test_game_over_detection():
    """Test game over is detected correctly."""
    game = Game2048()
    
    # Fill board with non-mergeable tiles
    game.board = np.array([
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [4, 2, 4, 2]
    ])
    
    # Try all moves - none should work
    game.move('up')
    assert not game.has_valid_moves()
    print("✓ Game over detection test passed")


def test_clone():
    """Test game cloning."""
    game = Game2048()
    game.score = 100
    
    clone = game.clone()
    assert np.array_equal(clone.board, game.board)
    assert clone.score == game.score
    
    # Modify clone - original should be unchanged
    clone.board[0, 0] = 999
    assert game.board[0, 0] != 999
    print("✓ Clone test passed")


def run_all_tests():
    """Run all tests."""
    print("Running 2048 game tests...\n")
    
    test_initialization()
    test_move_mechanics()
    test_merge_scoring()
    test_game_over_detection()
    test_clone()
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    run_all_tests()
