"""
Benchmark script for 2048 AI players.

Runs both the Expectimax AI and the N-Tuple AI (if ntuple.bin exists)
and prints per-game results plus summary statistics.
"""

import time

from ai_player import play_game_with_ai


def _print_summary(label: str, scores: list, max_tiles: list,
                   moves_list: list, elapsed: float) -> None:
    n = len(scores)
    print(f"\n{'='*48}")
    print(f"  {label} — {n} games")
    print(f"{'='*48}")
    print(f"  Average score : {sum(scores)/n:>10.0f}")
    print(f"  Best score    : {max(scores):>10,}")
    print(f"  Worst score   : {min(scores):>10,}")
    print(f"\n  Max tiles achieved:")
    for tile_val in sorted(set(max_tiles), reverse=True):
        count = max_tiles.count(tile_val)
        pct   = 100 * count / n
        print(f"    {tile_val:>6}: {count:>3} games  ({pct:.1f}%)")
    if moves_list:
        print(f"\n  Avg moves/game : {sum(moves_list)/n:.0f}")
    print(f"  Total time     : {elapsed:.1f}s  ({elapsed/n:.1f}s/game)")
    print(f"{'='*48}\n")


def run_benchmark(num_games: int = 10, search_depth: int = 3) -> None:
    """Benchmark the Expectimax AI."""
    print(f"\n=== Expectimax AI Benchmark  (depth={search_depth}) ===")
    scores, max_tiles, moves_list = [], [], []
    t0 = time.time()
    for i in range(num_games):
        print(f"  Game {i+1}/{num_games}…", end=" ", flush=True)
        score, max_tile, num_moves = play_game_with_ai(
            search_depth=search_depth, verbose=False
        )
        scores.append(score)
        max_tiles.append(max_tile)
        moves_list.append(num_moves)
        print(f"score={score:,}  max={max_tile}  moves={num_moves}")
    _print_summary("Expectimax AI", scores, max_tiles, moves_list,
                   time.time() - t0)


def run_ntuple_benchmark(num_games: int = 100,
                         weights_path: str | None = None) -> None:
    """Benchmark the N-Tuple TD AI."""
    from ntuple_network import board
    from ntuple_player import NTuplePlayer, WEIGHTS_PATH as _DEFAULT_WEIGHTS
    from game2048 import Game2048

    path = weights_path or _DEFAULT_WEIGHTS

    if not NTuplePlayer.is_available(path):
        print(f"\n[ntuple] No weights found at {path}")
        print("  Run  python td_trainer.py  first to train the network.\n")
        return

    print(f"\n=== N-Tuple AI Benchmark  ({num_games} games) ===")
    print(f"  Weights: {path}")

    print("  Loading weights…", end=" ", flush=True)
    player = NTuplePlayer(path)
    print("done.\n")

    scores, max_tiles = [], []
    t0 = time.time()

    for i in range(num_games):
        print(f"  Game {i+1}/{num_games}…", end=" ", flush=True)
        game = Game2048()
        while not game.game_over:
            direction = player.get_best_move(game)
            if direction is None:
                break
            game.move(direction)
        scores.append(game.score)
        max_tiles.append(game.get_max_tile())
        print(f"score={game.score:,}  max={game.get_max_tile()}")

    _print_summary("N-Tuple AI", scores, max_tiles, [], time.time() - t0)


if __name__ == "__main__":
    # Expectimax: quick sanity check (10 games at depth 3)
    run_benchmark(num_games=10, search_depth=3)

    # N-Tuple: 100-game benchmark (skipped automatically if ntuple.bin missing)
    run_ntuple_benchmark(num_games=100)
