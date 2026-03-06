"""
TD(0) trainer for the 2048 N-Tuple network.

Saves weights to ntuple.bin (resume-safe: run again to continue training).
Prints statistics every --unit games to stdout and optionally to TensorBoard.

Usage:
    python td_trainer.py                      # 100k games, alpha=0.1
    python td_trainer.py --total 500000       # 500k games
    python td_trainer.py --alpha 0.05         # custom learning rate
    python td_trainer.py --weights my.bin     # custom weights file
    python td_trainer.py --unit 500           # print stats every 500 games

Expected performance (from TDL2048-Demo reference):
    100k games  →  avg ~68,000  |  2048 win rate ~91%  |  4096 win rate ~69%
"""
import os
import sys
import signal
import argparse
import time
from datetime import datetime

from ntuple_network import board, pattern, learning

# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False

HERE = os.path.dirname(os.path.abspath(__file__))

# Standard 4×6-tuple patterns from TDL2048-Demo (Szubert & Jaśkowski 2014)
_PATTERNS = [
    [0, 1, 2, 3, 4, 5],    # rows 0+1, left 3 columns
    [4, 5, 6, 7, 8, 9],    # rows 1+2, left 3 columns
    [0, 1, 2, 4, 5, 6],    # 2×3 block, top-left
    [4, 5, 6, 8, 9, 10],   # 2×3 block, middle-left
]


def make_network() -> learning:
    tdl = learning()
    for patt in _PATTERNS:
        tdl.add_feature(pattern(patt))
    return tdl


def _print_stats(n: int, scores: list, maxtiles: list, unit: int,
                 elapsed: float, writer=None) -> None:
    avg       = sum(scores) / len(scores)
    best      = max(scores)
    rate      = unit / elapsed if elapsed > 0 else 0.0
    tile_cnt  = [maxtiles.count(t) for t in range(16)]

    print(f"{n:>10,}  avg = {avg:>10.1f}  max = {best:>8,}  ({rate:.0f} games/s)")

    shown = False
    for t in range(15, 0, -1):
        if tile_cnt[t] > 0:
            shown = True
        if shown:
            accu     = sum(tile_cnt[t:])
            tile_val = (1 << t) & -2  # actual tile value (e.g. 2048), mask out 1
            win_pct  = accu * 100 / unit
            own_pct  = tile_cnt[t] * 100 / unit
            print(f"          {tile_val:>6}  {win_pct:5.1f}%  ({own_pct:.1f}%)")
    print()

    if writer is not None:
        writer.add_scalar('train/avg_score',  avg,  n)
        writer.add_scalar('train/max_score',  best, n)
        writer.add_scalar('train/games_per_sec', rate, n)
        for t in range(4, 15):
            tile_val = (1 << t) & -2
            pct = sum(tile_cnt[t:]) * 100 / unit
            writer.add_scalar(f'tiles/pct_{tile_val}', pct, n)


def train(total: int, alpha: float, weights_path: str, unit: int = 1000) -> None:
    # ── Initialise lookup table and network ───────────────────────────────
    print("\nInitialising lookup table…")
    board.lookup.init()
    print("Done.\n")

    print("Network features:")
    tdl = make_network()
    print()

    resumed = tdl.load(weights_path)

    # ── TensorBoard ───────────────────────────────────────────────────────
    writer = None
    if _TB_AVAILABLE:
        run_name  = f"2048_tdl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_dir   = os.path.join(HERE, 'runs', run_name)
        writer    = SummaryWriter(log_dir)
        tb_cmd    = f"tensorboard --logdir \"{os.path.join(HERE, 'runs')}\""
        print(f"  TensorBoard : {tb_cmd}")
        print(f"  Run name    : {run_name}")
        print()

    # ── Header ────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  2048 N-Tuple TD Trainer")
    print("=" * 60)
    print(f"  Total games : {total:,}")
    print(f"  Alpha       : {alpha}")
    print(f"  Weights     : {weights_path}")
    print(f"  Resumed     : {'yes' if resumed else 'no — fresh start'}")
    print("=" * 60)
    print()

    # ── Graceful Ctrl+C ───────────────────────────────────────────────────
    interrupted = False

    def _on_interrupt(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\n  Ctrl+C — finishing current game then saving…\n")

    signal.signal(signal.SIGINT, _on_interrupt)

    # ── Training loop ─────────────────────────────────────────────────────
    scores: list[int]   = []
    maxtiles: list[int] = []
    t_chunk = time.time()

    for n in range(1, total + 1):
        if interrupted:
            break

        path:  list  = []
        state: board = board()
        score: int   = 0

        state.init()

        while True:
            best = tdl.select_best_move(state)
            path.append(best)
            if not best.is_valid():
                break
            score += best.reward()
            state  = board(best.afterstate())
            state.popup()

        tdl.learn_from_episode(path, alpha)

        scores.append(score)
        maxtiles.append(state.max_tile())

        if n % unit == 0:
            elapsed   = time.time() - t_chunk
            _print_stats(n, scores, maxtiles, unit, elapsed, writer)
            scores.clear()
            maxtiles.clear()
            t_chunk = time.time()

    # ── Save ──────────────────────────────────────────────────────────────
    print("Saving weights…")
    tdl.save(weights_path)

    if writer is not None:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train 2048 N-Tuple TD network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--total',   type=int,   default=100_000,
                        help='Total games to train')
    parser.add_argument('--alpha',   type=float, default=0.1,
                        help='TD learning rate')
    parser.add_argument('--weights', type=str,
                        default=os.path.join(HERE, 'ntuple.bin'),
                        help='Path to weights file (created / resumed)')
    parser.add_argument('--unit',    type=int,   default=1000,
                        help='Print progress stats every N games')
    args = parser.parse_args()

    train(
        total        = args.total,
        alpha        = args.alpha,
        weights_path = args.weights,
        unit         = args.unit,
    )


if __name__ == '__main__':
    main()
