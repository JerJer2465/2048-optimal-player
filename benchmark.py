"""
Benchmark script for 2048 AI player.
Tests performance across multiple games.
"""

from ai_player import play_game_with_ai
import time


def run_benchmark(num_games: int = 10, search_depth: int = 3):
    """Run multiple games and report statistics.
    
    Args:
        num_games: Number of games to play
        search_depth: AI search depth
    """
    print(f"=== 2048 AI Benchmark ===")
    print(f"Games: {num_games}")
    print(f"Search depth: {search_depth}")
    print(f"{'='*40}\n")
    
    scores = []
    max_tiles = []
    moves = []
    
    start_time = time.time()
    
    for i in range(num_games):
        print(f"Game {i+1}/{num_games}...", end=" ", flush=True)
        score, max_tile, num_moves = play_game_with_ai(
            search_depth=search_depth,
            verbose=False
        )
        scores.append(score)
        max_tiles.append(max_tile)
        moves.append(num_moves)
        print(f"Score: {score}, Max: {max_tile}, Moves: {num_moves}")
    
    elapsed = time.time() - start_time
    
    # Print statistics
    print(f"\n{'='*40}")
    print("RESULTS:")
    print(f"{'='*40}")
    print(f"Average score: {sum(scores)/len(scores):.0f}")
    print(f"Best score: {max(scores)}")
    print(f"Worst score: {min(scores)}")
    print(f"\nMax tiles achieved:")
    for tile_val in sorted(set(max_tiles), reverse=True):
        count = max_tiles.count(tile_val)
        pct = 100 * count / len(max_tiles)
        print(f"  {tile_val}: {count} games ({pct:.1f}%)")
    print(f"\nAverage moves per game: {sum(moves)/len(moves):.0f}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/num_games:.1f}s per game)")
    print(f"{'='*40}")


if __name__ == "__main__":
    # Quick benchmark: 10 games at depth 3
    run_benchmark(num_games=10, search_depth=3)
