# 2048 Optimal Player

AI player for the 2048 puzzle game with three AI backends, from strongest to weakest:

| AI | Avg Score | 2048 Win Rate | Notes |
|---|---|---|---|
| **N-Tuple TD** | ~68,000 | ~91% | After 100k training games (~1.8h CPU) |
| Expectimax | ~7,000 | ~90–95% | No training required, heuristic search |
| Neural (PPO) | ~1,075 | 0% | Work in progress — structurally limited |

## Project Structure

```
2048-optimal-player/
├── game2048.py          # Core 2048 game logic
├── game_ui.py           # Pygame UI — auto-selects best available AI
├── benchmark.py         # Performance benchmarking
├── test_game.py         # Unit tests
│
├── ntuple_network.py    # N-Tuple TD: bitboard, pattern features, learning
├── ntuple_player.py     # N-Tuple player UI adapter
├── td_trainer.py        # TD(0) self-play trainer
│
├── ai_player.py         # Expectimax AI (heuristic fallback)
│
├── game2048_env.py      # Gymnasium environment wrapper (for PPO)
├── neural_player.py     # CNN policy/value network + inference
├── train_neural.py      # PPO self-play trainer
│
├── requirements.txt     # Python dependencies
└── README.md
```

---

## Quick start

```bash
git clone https://github.com/JerJer2465/2048-optimal-player.git
cd 2048-optimal-player
pip3 install -r requirements.txt
python3 game_ui.py          # watch Expectimax AI (N-Tuple kicks in after training)
```

**Controls:**
| Key | Action |
|-----|--------|
| `SPACE` | Pause / resume |
| `R` | Reload checkpoint and start new game |
| `Q` | Quit |

The UI priority is: **N-Tuple AI** (orange) > Neural AI (green) > Expectimax (blue).

---

## N-Tuple TD Learning (primary AI)

Based on [Szubert & Jaśkowski, CIG 2014](https://doi.org/10.1109/CIG.2014.6932907) via
[moporgic/TDL2048-Demo](https://github.com/moporgic/TDL2048-Demo).

### How it works

The board is stored as a 64-bit bitboard (each of the 16 cells stores its log₂ tile
value in 4 bits). A **4×6-tuple network** is a collection of lookup tables, one per
board pattern:

```
Patterns (each with 8 isomorphisms = rotate + reflect):
  [0,1,2,3,4,5]   [4,5,6,7,8,9]   [0,1,2,4,5,6]   [4,5,6,8,9,10]
   . . . .          . . . .          . . . .          . . . .
   X X X X          X X X X          X X X .          . X X X
   X X . .          X X . .          X X . .          . X X X
   . . . .          . . . .          . . . .          . . . .

Total weight table: 4 × 16^6 × 4 bytes ≈ 256 MB
```

**Training** uses **TD(0) afterstate learning** — the key insight:

```
state_t  →  action_a  →  afterstate_ŝ_t  →  random tile  →  state_{t+1}
```

By learning `V(afterstate)` instead of `V(state)`, the TD target is fully deterministic
(no averaging over random tile spawns). The backward update per episode:

```python
target = 0
for each move (last → first):
    error  = target - V(afterstate)
    target = reward + update(afterstate, alpha × error)
```

### Train

```bash
python3 td_trainer.py                    # 100k games, saves ntuple.bin
python3 td_trainer.py --total 500000     # more training (resumes automatically)
python3 td_trainer.py --alpha 0.05       # lower learning rate
```

Progress prints every 1000 games:

```
     1,000  avg =    5,432.0  max =   18,234  (28 games/s)
    10,000  avg =   28,110.0  max =   95,600  (11 games/s)
   100,000  avg =   68,663.7  max =  177,508
               2048   91.2%  (22.5%)
               4096   68.7%  (53.9%)
               8192   14.8%  (14.8%)
```

*(Games/sec slows as the AI improves and games get longer.)*

### Monitor with TensorBoard

```bash
tensorboard --logdir runs
# open http://localhost:6006
```

### Benchmark

```bash
python3 benchmark.py    # runs Expectimax + N-Tuple benchmarks
```

### Expected milestones

| Games trained | Avg score | Notable |
|---|---|---|
| 1,000 | ~5,000 | Already 5× better than PPO after 12M steps |
| 10,000 | ~28,000 | Regularly reaching 1024 tile |
| 100,000 | ~68,000 | 91% 2048 win rate, 69% 4096 win rate |
| 500,000+ | ~120,000+ | 4096 consistent |

---

## Expectimax AI (fallback)

Used automatically when `ntuple.bin` is not found.

- **Algorithm**: Expectimax with iterative deepening (depth 3→5)
- **Transposition table**: ~60–80% cache hit rate
- **Heuristics**: monotonicity, smoothness, empty cells, corner bonus
- **2048 win rate**: ~90–95%

```bash
python3 ai_player.py    # run standalone (no UI)
```

---

## Neural Network / PPO (experimental)

A CNN trained with Proximal Policy Optimization. Left in place for research purposes —
PPO without lookahead is structurally unsuited to 2048 (long credit-assignment horizon,
exponential reward scale, sparse signal for large tiles).

```bash
caffeinate -i nohup python3 -u train_neural.py > training.log 2>&1 &
tail -f training.log
pkill -f train_neural.py    # stop
```

**Architecture** — shared CNN trunk with policy and value heads:

```
Input: (16, 4, 4) one-hot encoded board
  └─ Conv2d(16→128, 3×3) → BN → ReLU  ×3
  └─ Flatten → Linear(2048→512) → ReLU
        ├─ Policy head: Linear(512→4)
        └─ Value head:  Linear(512→1)
```

---

## License

MIT License
