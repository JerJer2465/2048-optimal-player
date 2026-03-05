"""
AlphaZero-style trainer for the 2048 neural network player.

Training has two phases:

  Phase 1 — Behaviour Cloning (BC)
    Play ~BC_GAMES games with the Expectimax AIPlayer (depth 3) and train
    the NN *policy head* to imitate its moves (cross-entropy).  This warms
    up the network with basic strategy (corner, monotonicity) in minutes
    instead of hours of random exploration.

  Phase 2 — Self-play Policy Iteration
    Repeatedly:
      (a) Generate SELFPLAY_GAMES_PER_ITER games using NeuralExpectimaxPlayer
          (depth-3 Expectimax with the NN value head at leaves).
      (b) Compute discounted MC returns for every step in every game.
      (c) Train BOTH heads for TRAIN_EPOCHS_PER_ITER epochs:
            policy_loss = CrossEntropy(logits, search_action)
            value_loss  = MSE(value_head, mc_return)
    As the value head improves, the search improves, which produces
    better self-play games, which produce better training targets —
    the AlphaZero virtuous cycle.

Run standalone (background-friendly):
    python train_neural.py
    caffeinate -i nohup python3 -u train_neural.py > training.log 2>&1 &

TensorBoard:
    tensorboard --logdir runs
    (open http://localhost:6006)

Stop:
    pkill -f train_neural.py
"""

import os
import sys
import signal
import time
import datetime
from collections import deque, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game2048_env import encode_board, DIRECTIONS, N_ACTIONS, OBS_SHAPE
from game2048 import Game2048
from neural_player import TwoFortyEightNet, NeuralExpectimaxPlayer, CHECKPOINT_PATH

# ── Hyperparameters ───────────────────────────────────────────────────────────

# Phase 1 — Behaviour Cloning
BC_GAMES       = 100    # Expectimax games to generate BC data (~50k states)
BC_EPOCHS      = 20     # supervised epochs over BC data
BC_BATCH_SIZE  = 512
BC_LR          = 3e-4

# Phase 2 — Self-play policy iteration
SEARCH_DEPTH            = 3     # Expectimax depth (play & self-play)
SELFPLAY_GAMES_PER_ITER = 50    # games generated per iteration
TRAIN_EPOCHS_PER_ITER   = 10    # training epochs over each self-play batch
TRAIN_BATCH_SIZE        = 512
TRAIN_LR                = 1e-4
GAMMA                   = 0.99  # discount for MC returns
MAX_GRAD_NORM           = 0.5
POLICY_LOSS_WEIGHT      = 1.0
VALUE_LOSS_WEIGHT       = 1.0

TOTAL_ITERS             = 2000  # self-play iterations (~hours of training)
CHECKPOINT_EVERY_ITERS  = 5     # save checkpoint every N iterations
LOG_EVERY_ITERS         = 1     # print stats every N iterations

# ─────────────────────────────────────────────────────────────────────────────


# ── Data generation ───────────────────────────────────────────────────────────

def generate_bc_data(n_games: int, ai_search_depth: int = 3):
    """Play games with the Expectimax AIPlayer and record (obs, action) pairs.

    Only the policy head is trained during BC; the value head is left to
    self-play to avoid scale-mismatch issues with the heuristic signal.

    Returns
    -------
    obs     : float32 array  (N, 16, 4, 4)
    actions : int64 array    (N,)
    """
    from ai_player import AIPlayer

    ai       = AIPlayer(search_depth=ai_search_depth)
    obs_buf  = []
    act_buf  = []

    print(f"  Generating BC data from {n_games} AIPlayer games (depth {ai_search_depth})…")
    t0 = time.time()

    for g in range(n_games):
        game = Game2048()
        while not game.game_over:
            move = ai.get_best_move(game)
            if move is None:
                break
            obs_buf.append(encode_board(game.board))
            act_buf.append(DIRECTIONS.index(move))
            game.move(move)

        if (g + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"    {g+1}/{n_games} games, {len(obs_buf):,} states  "
                  f"({elapsed:.0f}s)")

    print(f"  BC dataset: {len(obs_buf):,} states from {n_games} games  "
          f"({time.time()-t0:.0f}s total)")

    return (
        np.array(obs_buf, dtype=np.float32),
        np.array(act_buf,  dtype=np.int64),
    )


def generate_selfplay_games(player: NeuralExpectimaxPlayer, n_games: int):
    """Play games with NeuralExpectimaxPlayer; return per-step training data.

    For each game, MC returns are computed backwards so the value head
    learns to predict expected future log-reward.

    Returns
    -------
    obs     : float32 (N, 16, 4, 4)
    actions : int64   (N,)
    returns : float32 (N,)   discounted MC returns
    scores  : list of int    final game scores
    max_tiles: list of int   final max tiles
    """
    all_obs, all_actions, all_returns = [], [], []
    scores, max_tiles = [], []

    for _ in range(n_games):
        game     = Game2048()
        ep_obs   = []
        ep_acts  = []
        ep_rew   = []

        while not game.game_over:
            obs  = encode_board(game.board)
            move = player.get_best_move(game)
            if move is None:
                break

            action     = DIRECTIONS.index(move)
            prev_score = game.score
            moved      = game.move(move)
            delta      = game.score - prev_score

            reward = float(np.log2(delta + 1)) if (moved and delta > 0) else (-1.0 if not moved else 0.0)

            ep_obs.append(obs)
            ep_acts.append(action)
            ep_rew.append(reward)

        mc_returns = _mc_returns(ep_rew, GAMMA)

        all_obs.extend(ep_obs)
        all_actions.extend(ep_acts)
        all_returns.extend(mc_returns)
        scores.append(game.score)
        max_tiles.append(game.get_max_tile())

    return (
        np.array(all_obs,     dtype=np.float32),
        np.array(all_actions, dtype=np.int64),
        np.array(all_returns, dtype=np.float32),
        scores,
        max_tiles,
    )


def _mc_returns(rewards: list, gamma: float) -> np.ndarray:
    """Compute discounted returns R_t = r_t + γ·R_{t+1} backwards."""
    n       = len(rewards)
    returns = np.zeros(n, dtype=np.float32)
    R       = 0.0
    for t in reversed(range(n)):
        R         = rewards[t] + gamma * R
        returns[t] = R
    return returns


# ── Training step ──────────────────────────────────────────────────────────────

def supervised_update(
    model: TwoFortyEightNet,
    optimizer: torch.optim.Optimizer,
    obs: np.ndarray,
    actions: np.ndarray,
    returns,          # np.ndarray | None  — None means BC phase (policy head only)
    n_epochs: int,
    batch_size: int,
    device: torch.device,
) -> tuple:
    """Supervised update on a dataset of (obs, action[, mc_return]) triples.

    During BC (returns=None) only the policy head is trained.
    During self-play both heads are trained.

    Returns
    -------
    mean policy loss, mean value loss (0.0 when returns is None)
    """
    n     = len(obs)
    obs_t = torch.tensor(obs,     dtype=torch.float32, device=device)
    act_t = torch.tensor(actions, dtype=torch.long,    device=device)
    ret_t = (torch.tensor(returns, dtype=torch.float32, device=device)
             if returns is not None else None)

    p_losses, v_losses = [], []

    model.train()
    for _ in range(n_epochs):
        idx = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            b              = idx[start : start + batch_size]
            logits, values = model(obs_t[b])

            p_loss = F.cross_entropy(logits, act_t[b])
            v_loss = (F.mse_loss(values.squeeze(1), ret_t[b])
                      if ret_t is not None
                      else torch.tensor(0.0, device=device))

            loss = POLICY_LOSS_WEIGHT * p_loss + VALUE_LOSS_WEIGHT * v_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            p_losses.append(p_loss.item())
            v_losses.append(v_loss.item())

    return float(np.mean(p_losses)), float(np.mean(v_losses))


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, iteration, episodes_done,
                    recent_scores, recent_max_tiles, best_score,
                    bc_done: bool = False):
    data = {
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'total_steps':          int(iteration),
        'episodes_done':        int(episodes_done),
        'avg_score_recent':     float(np.mean(recent_scores))    if recent_scores else 0.0,
        'avg_max_tile_recent':  float(np.mean(recent_max_tiles)) if recent_max_tiles else 0.0,
        'best_score':           int(best_score),
        'bc_done':              bool(bc_done),
    }
    tmp = CHECKPOINT_PATH + '.tmp'
    torch.save(data, tmp)
    os.replace(tmp, CHECKPOINT_PATH)   # atomic on POSIX


# ── TensorBoard helpers ───────────────────────────────────────────────────────

def tile_histogram(tiles) -> str:
    counts = Counter(int(t) for t in tiles)
    parts  = [f"{tile}:{100*c/len(tiles):.0f}%"
              for tile, c in sorted(counts.items())]
    return "  ".join(parts)


def log_tile_pcts(writer: SummaryWriter, tiles, step: int):
    if not tiles:
        return
    n = len(tiles)
    for thr in [256, 512, 1024, 2048, 4096, 8192]:
        pct = 100.0 * sum(t >= thr for t in tiles) / n
        writer.add_scalar(f"tiles/pct_reached_{thr}", pct, step)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    run_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "runs",
        f"2048_az_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    writer = SummaryWriter(log_dir=run_dir)

    print("=" * 65)
    print("  2048 Neural Network — AlphaZero-style Trainer")
    print("=" * 65)
    print(f"  Device          : {device}")
    print(f"  Search depth    : {SEARCH_DEPTH}")
    print(f"  BC games        : {BC_GAMES}")
    print(f"  Self-play iters : {TOTAL_ITERS}  "
          f"({SELFPLAY_GAMES_PER_ITER} games/iter)")
    print(f"  Checkpoint      : {CHECKPOINT_PATH}")
    print(f"  TensorBoard     : {run_dir}")
    print(f"  Dashboard       : tensorboard --logdir runs")
    print(f"  Stop            : pkill -f train_neural.py")
    print("=" * 65)
    print()

    model     = TwoFortyEightNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=BC_LR)

    # ── Resume from checkpoint ────────────────────────────────────────────
    start_iter     = 0
    episodes_done  = 0
    best_score     = 0
    bc_done        = False   # skip BC if resuming after it completed

    if os.path.exists(CHECKPOINT_PATH):
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_iter    = int(ckpt.get('total_steps', 0))
            episodes_done = int(ckpt.get('episodes_done', 0))
            best_score    = int(ckpt.get('best_score', 0))
            bc_done       = bool(ckpt.get('bc_done', False))
            print(f"Resumed from checkpoint: iter {start_iter}, "
                  f"{episodes_done} episodes, best {best_score:,}")
            if bc_done:
                print("  (BC phase already completed — skipping to self-play)")
            print()
        except Exception as exc:
            print(f"Could not load checkpoint ({exc}), starting fresh.\n")

    # ── Graceful shutdown ─────────────────────────────────────────────────
    shutdown = [False]

    def _sig(sig, frame):
        print("\nShutdown requested — finishing current iteration…")
        shutdown[0] = True

    signal.signal(signal.SIGINT,  _sig)
    signal.signal(signal.SIGTERM, _sig)

    recent_scores    = deque(maxlen=500)
    recent_max_tiles = deque(maxlen=500)

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 1 — Behaviour Cloning
    # ─────────────────────────────────────────────────────────────────────
    if not bc_done and not shutdown[0]:
        print("─" * 65)
        print("PHASE 1  Behaviour Cloning")
        print("─" * 65)

        bc_obs, bc_acts = generate_bc_data(BC_GAMES, ai_search_depth=3)

        print(f"\nTraining policy head on {len(bc_obs):,} states "
              f"for {BC_EPOCHS} epochs…")
        t0 = time.time()

        # Policy-only update (returns=None → value head skipped)
        p_loss, _ = supervised_update(
            model, optimizer, bc_obs, bc_acts,
            returns=None,
            n_epochs=BC_EPOCHS,
            batch_size=BC_BATCH_SIZE,
            device=device,
        )

        print(f"  BC complete — policy_loss={p_loss:.4f}  "
              f"({time.time()-t0:.0f}s)")

        # Save immediately so the UI can use the BC-warmed model.
        # bc_done=True ensures resume skips Phase 1.
        save_checkpoint(model, optimizer, 0, 0, [], [], 0, bc_done=True)
        writer.add_scalar("bc/policy_loss", p_loss, 0)
        print("  Checkpoint saved — UI now uses BC-warmed Neural+Search player.\n")
        bc_done = True

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 2 — Self-play policy iteration
    # ─────────────────────────────────────────────────────────────────────
    print("─" * 65)
    print("PHASE 2  Self-play policy iteration")
    print("─" * 65)
    print()

    # Switch to self-play LR
    for pg in optimizer.param_groups:
        pg['lr'] = TRAIN_LR

    start_time = time.time()

    for iteration in range(start_iter + 1, TOTAL_ITERS + 1):
        if shutdown[0]:
            break

        iter_t0 = time.time()

        # ── (a) Build NeuralExpectimaxPlayer with current model ───────────
        model.eval()
        player = NeuralExpectimaxPlayer(model, device, search_depth=SEARCH_DEPTH)

        # ── (b) Generate self-play games ──────────────────────────────────
        obs, actions, returns, scores, tiles = generate_selfplay_games(
            player, SELFPLAY_GAMES_PER_ITER
        )

        for s, t in zip(scores, tiles):
            recent_scores.append(s)
            recent_max_tiles.append(t)
            best_score = max(best_score, s)
            episodes_done += 1

        # ── (c) Train both heads ──────────────────────────────────────────
        p_loss, v_loss = supervised_update(
            model, optimizer, obs, actions,
            returns=returns,
            n_epochs=TRAIN_EPOCHS_PER_ITER,
            batch_size=TRAIN_BATCH_SIZE,
            device=device,
        )

        iter_elapsed = time.time() - iter_t0
        total_elapsed = time.time() - start_time

        # ── TensorBoard ───────────────────────────────────────────────────
        writer.add_scalar("losses/policy_loss", p_loss,   iteration)
        writer.add_scalar("losses/value_loss",  v_loss,   iteration)
        writer.add_scalar("train/episodes",     episodes_done, iteration)

        if iteration % LOG_EVERY_ITERS == 0 and recent_scores:
            avg_score  = float(np.mean(recent_scores))
            med_score  = float(np.median(recent_scores))
            p25        = float(np.percentile(recent_scores, 25))
            p75        = float(np.percentile(recent_scores, 75))
            avg_tile   = float(np.mean(recent_max_tiles))
            eta_s      = max(0, (TOTAL_ITERS - iteration) * iter_elapsed)
            eta_h      = eta_s / 3600

            print(
                f"[iter {iteration:>5}]  "
                f"ep={episodes_done:>6,}  "
                f"avg={avg_score:>8.0f}  "
                f"p25/p75={p25:.0f}/{p75:.0f}  "
                f"max={best_score:>8,}  "
                f"tile={avg_tile:>6.0f}  "
                f"p_loss={p_loss:.3f}  v_loss={v_loss:.3f}  "
                f"({iter_elapsed:.1f}s/iter  ETA={eta_h:.1f}h)"
            )
            if len(recent_max_tiles) >= 10:
                print(f"  tile dist: {tile_histogram(recent_max_tiles)}")

            writer.add_scalar("charts/avg_score",    avg_score,  iteration)
            writer.add_scalar("charts/median_score", med_score,  iteration)
            writer.add_scalar("charts/p25_score",    p25,        iteration)
            writer.add_scalar("charts/p75_score",    p75,        iteration)
            writer.add_scalar("charts/best_score",   best_score, iteration)
            writer.add_scalar("charts/avg_max_tile", avg_tile,   iteration)
            log_tile_pcts(writer, recent_max_tiles, iteration)

        if iteration % CHECKPOINT_EVERY_ITERS == 0:
            save_checkpoint(model, optimizer, iteration, episodes_done,
                            recent_scores, recent_max_tiles, best_score,
                            bc_done=True)
            print(f"  ✓ checkpoint saved  (iter {iteration})")

    # ── Final save ────────────────────────────────────────────────────────
    save_checkpoint(model, optimizer, TOTAL_ITERS, episodes_done,
                    recent_scores, recent_max_tiles, best_score, bc_done=True)
    writer.flush()
    writer.close()

    print()
    print("=" * 65)
    print("Training finished.")
    print(f"  Episodes        : {episodes_done:,}")
    if recent_scores:
        print(f"  Recent avg score: {np.mean(recent_scores):.0f}")
        print(f"  Best score      : {best_score:,}")
        print(f"  Tile dist (last {len(recent_max_tiles)}): "
              f"{tile_histogram(recent_max_tiles)}")
    print(f"  Checkpoint      : {CHECKPOINT_PATH}")
    print(f"  TensorBoard     : {run_dir}")
    print("=" * 65)


if __name__ == '__main__':
    main()
