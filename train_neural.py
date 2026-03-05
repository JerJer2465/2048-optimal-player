"""
PPO self-play trainer for the 2048 neural network player.

Run standalone (background-friendly):
    python train_neural.py
    python train_neural.py &          # detached shell
    nohup python train_neural.py &    # survives terminal close

Training saves a checkpoint to model_checkpoint.pth every
CHECKPOINT_EVERY_EPISODES completed episodes. The game UI picks it
up automatically between games.

TensorBoard logs are written to ./runs/2048_ppo_<timestamp>/.
Launch the dashboard in a separate terminal:
    tensorboard --logdir runs

Press Ctrl+C to stop; the checkpoint is saved before exit.
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
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game2048_env import Game2048Env, OBS_SHAPE, N_ACTIONS
from neural_player import TwoFortyEightNet, CHECKPOINT_PATH

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_ENVS = 8
N_STEPS = 512               # steps collected per env per rollout
BATCH_SIZE = N_ENVS * N_STEPS  # 4096 total transitions per update
MINI_BATCH_SIZE = 512
N_EPOCHS = 10               # PPO epochs per rollout

GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF_START = 0.05   # decays toward ENTROPY_COEF_END over training
ENTROPY_COEF_END = 0.005
VALUE_COEF = 0.5
LR_START = 3e-4
LR_END = 1e-5
MAX_GRAD_NORM = 0.5

TOTAL_STEPS = 30_000_000    # increase for longer training (~3–6 h on CPU)
CHECKPOINT_EVERY_EPISODES = 500
LOG_EVERY_EPISODES = 50
# ─────────────────────────────────────────────────────────────────────────────


class SimpleVecEnv:
    """Sequential vectorized environment — avoids multiprocessing complexity
    while still allowing batched neural-network inference over all envs."""

    def __init__(self, n_envs: int):
        self.envs = [Game2048Env() for _ in range(n_envs)]
        self.n_envs = n_envs

    def reset(self) -> np.ndarray:
        return np.array([env.reset()[0] for env in self.envs], dtype=np.float32)

    def step(self, actions: np.ndarray):
        obs_list, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, _, info = env.step(int(action))
            if terminated:
                obs, _ = env.reset()
            obs_list.append(obs)
            rewards.append(reward)
            dones.append(float(terminated))
            infos.append(info)
        return (
            np.array(obs_list, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            infos,
        )


class RolloutBuffer:
    """Stores one full rollout for all environments."""

    def __init__(self, n_steps: int, n_envs: int):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs = np.zeros((n_steps, n_envs, *OBS_SHAPE), dtype=np.float32)
        self.actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.advantages = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.returns = np.zeros((n_steps, n_envs), dtype=np.float32)

    def compute_gae(self, last_values: np.ndarray, last_dones: np.ndarray):
        """Generalized Advantage Estimation (Schulman et al. 2015)."""
        last_adv = np.zeros(self.n_envs, dtype=np.float32)
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_values = self.values[t + 1]
            delta = (
                self.rewards[t]
                + GAMMA * next_values * next_non_terminal
                - self.values[t]
            )
            last_adv = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_adv
            self.advantages[t] = last_adv
        self.returns = self.advantages + self.values

    def mini_batches(self, mini_batch_size: int):
        n = self.n_steps * self.n_envs
        idx = np.random.permutation(n)
        flat_obs = self.obs.reshape(n, *OBS_SHAPE)
        flat_act = self.actions.reshape(n)
        flat_lp = self.log_probs.reshape(n)
        flat_adv = self.advantages.reshape(n)
        flat_ret = self.returns.reshape(n)
        for start in range(0, n, mini_batch_size):
            b = idx[start : start + mini_batch_size]
            yield flat_obs[b], flat_act[b], flat_lp[b], flat_adv[b], flat_ret[b]


def ppo_update(model, optimizer, buffer, entropy_coef, device):
    """Run N_EPOCHS of PPO updates over the rollout buffer."""
    # Normalize advantages
    adv_flat = buffer.advantages.reshape(-1)
    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
    buffer.advantages = adv_flat.reshape(buffer.n_steps, buffer.n_envs)

    p_losses, v_losses, entropies = [], [], []

    for _ in range(N_EPOCHS):
        for obs_b, act_b, old_lp_b, adv_b, ret_b in buffer.mini_batches(MINI_BATCH_SIZE):
            obs_t = torch.tensor(obs_b, device=device)
            act_t = torch.tensor(act_b, dtype=torch.long, device=device)
            old_lp_t = torch.tensor(old_lp_b, device=device)
            adv_t = torch.tensor(adv_b, device=device)
            ret_t = torch.tensor(ret_b, device=device)

            logits, values = model(obs_t)
            dist = Categorical(logits=logits)
            new_lp = dist.log_prob(act_t)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_lp - old_lp_t)
            p_loss = -torch.min(
                ratio * adv_t,
                torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t,
            ).mean()

            v_loss = F.mse_loss(values.squeeze(1), ret_t)
            loss = p_loss + VALUE_COEF * v_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            p_losses.append(p_loss.item())
            v_losses.append(v_loss.item())
            entropies.append(entropy.item())

    return float(np.mean(p_losses)), float(np.mean(v_losses)), float(np.mean(entropies))


def save_checkpoint(model, optimizer, total_steps, episodes_done, recent_scores,
                    recent_max_tiles, best_score):
    stats = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'total_steps': total_steps,
        'episodes_done': episodes_done,
        'avg_score_recent': float(np.mean(recent_scores)) if recent_scores else 0.0,
        'avg_max_tile_recent': float(np.mean(recent_max_tiles)) if recent_max_tiles else 0.0,
        'best_score': best_score,
    }
    tmp_path = CHECKPOINT_PATH + '.tmp'
    torch.save(stats, tmp_path)
    os.replace(tmp_path, CHECKPOINT_PATH)  # atomic on POSIX


def linear_schedule(start: float, end: float, step: int, total: int) -> float:
    return start + min(step / total, 1.0) * (end - start)


def tile_histogram(tiles):
    """Return a compact string showing the distribution of max tiles."""
    counts = Counter(int(t) for t in tiles)
    parts = []
    for tile in sorted(counts):
        pct = 100.0 * counts[tile] / len(tiles)
        parts.append(f"{tile}:{pct:.0f}%")
    return "  ".join(parts)


def log_tile_percentages(writer: SummaryWriter, tiles, step: int):
    """Write the % of games that reached each tile threshold to TensorBoard."""
    if not tiles:
        return
    tiles_list = [int(t) for t in tiles]
    n = len(tiles_list)
    for threshold in [256, 512, 1024, 2048, 4096, 8192]:
        pct = 100.0 * sum(t >= threshold for t in tiles_list) / n
        writer.add_scalar(f"tiles/pct_reached_{threshold}", pct, step)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── TensorBoard writer ────────────────────────────────────────────────
    run_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "runs",
        f"2048_ppo_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    writer = SummaryWriter(log_dir=run_dir)

    print("=" * 65)
    print("  2048 Neural Network — PPO Self-Play Trainer")
    print("=" * 65)
    print(f"  Device        : {device}")
    print(f"  Environments  : {N_ENVS}")
    print(f"  Steps/rollout : {BATCH_SIZE:,}  ({N_STEPS} steps × {N_ENVS} envs)")
    print(f"  Total budget  : {TOTAL_STEPS:,} steps")
    print(f"  Checkpoint    : {CHECKPOINT_PATH}")
    print(f"  TensorBoard   : {run_dir}")
    print(f"  Dashboard     : tensorboard --logdir runs")
    print(f"  Stop training : Ctrl+C  (checkpoint is saved before exit)")
    print("=" * 65)
    print()

    model = TwoFortyEightNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_START)

    # ── Resume from existing checkpoint ──────────────────────────────────
    total_steps = 0
    episodes_done = 0
    best_score = 0
    if os.path.exists(CHECKPOINT_PATH):
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            total_steps = int(ckpt.get('total_steps', 0))
            episodes_done = int(ckpt.get('episodes_done', 0))
            best_score = int(ckpt.get('best_score', 0))
            print(
                f"Resumed: {total_steps:,} steps, "
                f"{episodes_done:,} episodes, best score {best_score:,}"
            )
            print()
        except Exception as exc:
            print(f"Could not load checkpoint ({exc}), starting fresh.\n")

    # ── Env + buffer ──────────────────────────────────────────────────────
    vec_env = SimpleVecEnv(N_ENVS)
    buffer = RolloutBuffer(N_STEPS, N_ENVS)
    obs = vec_env.reset()

    recent_scores: deque = deque(maxlen=500)
    recent_max_tiles: deque = deque(maxlen=500)
    episodes_since_ckpt = 0
    episodes_since_log = 0
    start_steps = total_steps
    start_time = time.time()

    # ── Graceful shutdown ─────────────────────────────────────────────────
    shutdown = [False]

    def _handle_signal(sig, frame):
        print("\nShutdown signal received — finishing current rollout …")
        shutdown[0] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Save an initial checkpoint immediately so the UI sees the model exists
    save_checkpoint(model, optimizer, total_steps, episodes_done,
                    recent_scores, recent_max_tiles, best_score)
    print("Initial checkpoint written.")
    print()

    # ── Training loop ─────────────────────────────────────────────────────
    rollout_num = 0
    while total_steps < TOTAL_STEPS and not shutdown[0]:
        rollout_num += 1

        # ── Collect rollout ───────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            for step in range(N_STEPS):
                if shutdown[0]:
                    break

                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
                logits, values = model(obs_t)
                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

                actions_np = actions.cpu().numpy()
                new_obs, rewards, dones, infos = vec_env.step(actions_np)

                buffer.obs[step] = obs
                buffer.actions[step] = actions_np
                buffer.rewards[step] = rewards
                buffer.values[step] = values.squeeze(1).cpu().numpy()
                buffer.log_probs[step] = log_probs.cpu().numpy()
                buffer.dones[step] = dones
                obs = new_obs
                total_steps += N_ENVS

                # Track completed episodes
                for done, info in zip(dones, infos):
                    if done:
                        score = int(info['score'])
                        max_tile = int(info['max_tile'])
                        recent_scores.append(score)
                        recent_max_tiles.append(max_tile)
                        best_score = max(best_score, score)
                        episodes_done += 1
                        episodes_since_ckpt += 1
                        episodes_since_log += 1

            # Bootstrap value for the last observation
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            _, last_values = model(obs_t)
            last_values_np = last_values.squeeze(1).cpu().numpy()

        buffer.compute_gae(last_values_np, buffer.dones[-1])

        # ── Adjust schedules ──────────────────────────────────────────────
        lr = linear_schedule(LR_START, LR_END, total_steps, TOTAL_STEPS)
        entropy_coef = linear_schedule(
            ENTROPY_COEF_START, ENTROPY_COEF_END, total_steps, TOTAL_STEPS
        )
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # ── PPO update ────────────────────────────────────────────────────
        model.train()
        p_loss, v_loss, ent = ppo_update(
            model, optimizer, buffer, entropy_coef, device
        )

        # ── TensorBoard — per-rollout losses and schedules ────────────────
        writer.add_scalar("losses/policy_loss",  p_loss,       total_steps)
        writer.add_scalar("losses/value_loss",   v_loss,       total_steps)
        writer.add_scalar("losses/entropy",      ent,          total_steps)
        writer.add_scalar("train/learning_rate", lr,           total_steps)
        writer.add_scalar("train/entropy_coef",  entropy_coef, total_steps)
        writer.add_scalar("train/episodes",      episodes_done, total_steps)

        # ── Logging ───────────────────────────────────────────────────────
        if episodes_since_log >= LOG_EVERY_EPISODES and recent_scores:
            elapsed = time.time() - start_time
            sps = (total_steps - start_steps) / max(elapsed, 1)
            eta_s = max(0, (TOTAL_STEPS - total_steps) / max(sps, 1))
            eta_h = eta_s / 3600

            avg_score = float(np.mean(recent_scores))
            avg_tile = float(np.mean(recent_max_tiles))
            med_score = float(np.median(recent_scores))
            p25_score = float(np.percentile(recent_scores, 25))
            p75_score = float(np.percentile(recent_scores, 75))

            print(
                f"[{total_steps:>10,}] "
                f"ep={episodes_done:>7,}  "
                f"avg={avg_score:>8.0f}  "
                f"max={best_score:>8,}  "
                f"tile={avg_tile:>6.0f}  "
                f"lr={lr:.1e}  ent={entropy_coef:.4f}  "
                f"sps={sps:>5.0f}  ETA={eta_h:.1f}h"
            )
            if len(recent_max_tiles) >= 20:
                print(f"  tile dist: {tile_histogram(recent_max_tiles)}")

            # TensorBoard — per-log-interval episode stats
            writer.add_scalar("charts/avg_score",      avg_score,  total_steps)
            writer.add_scalar("charts/median_score",   med_score,  total_steps)
            writer.add_scalar("charts/p25_score",      p25_score,  total_steps)
            writer.add_scalar("charts/p75_score",      p75_score,  total_steps)
            writer.add_scalar("charts/best_score",     best_score, total_steps)
            writer.add_scalar("charts/avg_max_tile",   avg_tile,   total_steps)
            writer.add_scalar("charts/steps_per_sec",  sps,        total_steps)
            log_tile_percentages(writer, recent_max_tiles, total_steps)

            episodes_since_log = 0

        # ── Checkpoint ────────────────────────────────────────────────────
        if episodes_since_ckpt >= CHECKPOINT_EVERY_EPISODES:
            save_checkpoint(
                model, optimizer, total_steps, episodes_done,
                recent_scores, recent_max_tiles, best_score
            )
            episodes_since_ckpt = 0
            print(f"  ✓ checkpoint saved  (step {total_steps:,})")

    # ── Final save ────────────────────────────────────────────────────────
    save_checkpoint(model, optimizer, total_steps, episodes_done,
                    recent_scores, recent_max_tiles, best_score)
    writer.flush()
    writer.close()
    print()
    print("=" * 65)
    print(f"Training finished.")
    print(f"  Total steps    : {total_steps:,}")
    print(f"  Total episodes : {episodes_done:,}")
    if recent_scores:
        print(f"  Recent avg score: {np.mean(recent_scores):.0f}")
        print(f"  Best score      : {best_score:,}")
        print(f"  Tile dist (last {len(recent_max_tiles)}): "
              f"{tile_histogram(recent_max_tiles)}")
    print(f"  Checkpoint saved to: {CHECKPOINT_PATH}")
    print(f"  TensorBoard logs  : {run_dir}")
    print("=" * 65)


if __name__ == '__main__':
    main()
