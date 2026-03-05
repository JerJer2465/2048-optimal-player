# 2048 Optimal Player

AI player for the 2048 puzzle game. Includes both a classic Expectimax AI and a **neural network player trained via PPO reinforcement learning** that targets 40,000+ scores consistently.

## Project Structure

```
2048-optimal-player/
├── game2048.py          # Core 2048 game logic
├── game2048_env.py      # Gymnasium environment wrapper (for RL training)
├── ai_player.py         # Expectimax AI (heuristic fallback)
├── neural_player.py     # CNN policy/value network + inference
├── train_neural.py      # PPO self-play trainer (runs in background)
├── game_ui.py           # Pygame UI — auto-selects Neural or Expectimax AI
├── benchmark.py         # Performance testing
├── test_game.py         # Unit tests
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Installation

```bash
git clone https://github.com/JerJer2465/2048-optimal-player.git
cd 2048-optimal-player
pip3 install -r requirements.txt
```

---

## Running the game

```bash
python3 game_ui.py
```

The UI automatically uses the **neural network** if a trained checkpoint exists, otherwise falls back to the Expectimax AI.

**Controls:**
| Key | Action |
|-----|--------|
| `SPACE` | Pause / resume |
| `R` | Reload latest checkpoint and start a new game |
| `Q` | Quit |

---

## Training the neural network

### Start training (keeps running while lid is closed)

```bash
cd /path/to/2048-optimal-player
caffeinate -i nohup python3 train_neural.py > training.log 2>&1 &
```

- `caffeinate -i` — prevents macOS from sleeping (training continues with lid closed)
- `nohup` — keeps the process alive if the terminal is closed
- Output is saved to `training.log`

### Monitor training output

```bash
tail -f training.log
```

### Stop training

```bash
pkill -f train_neural.py
```

### TensorBoard (real-time metrics dashboard)

```bash
tensorboard --logdir runs
# then open http://localhost:6006 in your browser
```

Logged metrics include:
- `charts/avg_score`, `charts/median_score`, `charts/p25_score`, `charts/p75_score`
- `charts/best_score`, `charts/avg_max_tile`
- `tiles/pct_reached_1024`, `tiles/pct_reached_2048`, `tiles/pct_reached_4096`, …
- `losses/policy_loss`, `losses/value_loss`, `losses/entropy`
- `train/learning_rate`, `train/entropy_coef`

Each training run writes to its own timestamped subfolder under `runs/`, so you can compare runs side-by-side.

### Expected training timeline (CPU)

| Training time | Avg score | Notes |
|---|---|---|
| 0 min | — | Falls back to Expectimax AI (~7k avg) |
| 5–10 min | First checkpoint | Neural player active in UI |
| 1–2 hours | ~40,000+ | Consistently reaches 4096 tile |
| 4–6 hours | ~100,000+ | Approaches 8192 tile on good runs |

Training resumes automatically from the latest checkpoint if you stop and restart.

---

## How the neural network works

**Architecture** — shared-trunk CNN with a policy head and a value head:

```
Input: (16, 4, 4) one-hot encoded board
       (channel 0 = empty, channel k = tile 2^k)
  └─ Conv2d(16→128, 3×3) → BatchNorm → ReLU
  └─ Conv2d(128→128, 3×3) → BatchNorm → ReLU
  └─ Conv2d(128→128, 3×3) → BatchNorm → ReLU
  └─ Flatten → Linear(2048→512) → ReLU
        ├─ Policy head: Linear(512→4)   → action probabilities
        └─ Value head:  Linear(512→1)   → state value estimate
```

**Training** — PPO (Proximal Policy Optimization) self-play with shaped rewards:
- `log2(score_delta + 1)` per step — continuous, never sparse
- `+100 × log2(new_max_tile)` bonus whenever a new max tile is reached
- `-1` for invalid moves (board unchanged)

---

## Classic Expectimax AI

Used as a fallback when no checkpoint exists.

- **Algorithm**: Expectimax with iterative deepening (depth 3→5)
- **Transposition table**: ~60–80% cache hit rate
- **Heuristics**: monotonicity, smoothness, empty cells, corner strategy
- **Win rate (2048 tile)**: ~90–95%

```bash
# Run Expectimax only (no UI)
python3 ai_player.py

# Benchmark Expectimax (10 games)
python3 benchmark.py
```

---

## License

MIT License
