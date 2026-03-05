# 2048 Optimal Player

AI player for the 2048 puzzle game with optimal strategy implementation.

## Project Structure

```
2048-optimal-player/
├── game2048.py          # Core 2048 game logic
├── ai_player.py         # AI player (Expectimax + heuristics)
├── benchmark.py         # Performance testing
├── test_game.py         # Test script
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Current Status

✅ **Phase 1: Game Implementation (COMPLETE)**
- Core 2048 game logic
- Move operations (up, down, left, right)
- Tile merging and scoring
- Game over detection
- Random tile spawning

✅ **Phase 2: AI Player (COMPLETE)**
- Expectimax algorithm with configurable search depth
- Multi-heuristic evaluation:
  - Monotonicity (increasing/decreasing tile sequences)
  - Smoothness (similar adjacent tiles)
  - Empty cells (maximize available moves)
  - Corner strategy (keep max tile in corner)
- Optimal move selection to maximize score

## Installation

```bash
# Clone the repository
git clone https://github.com/JerJer2465/2048-optimal-player.git
cd 2048-optimal-player

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Play a random game (demo)

```bash
python game2048.py
```

### Test the game logic

```bash
python test_game.py
```

### Watch the AI play

```bash
python ai_player.py
```

### Benchmark AI performance

```bash
python benchmark.py
```

The benchmark will run 10 games and report:
- Average/best/worst scores
- Max tile distribution (how often it reaches 512, 1024, 2048, etc.)
- Average moves per game
- Time per game

## Game Features

- **4x4 board** (configurable size)
- **Tile spawning**: 90% chance of 2, 10% chance of 4
- **Move validation**: Only valid moves are executed
- **Score tracking**: Points awarded for tile merges
- **Game over detection**: Automatically detects when no moves remain

## Next Steps

1. Implement Expectimax algorithm for optimal play
2. Add heuristic evaluation (monotonicity, smoothness, empty cells)
3. Add performance benchmarking
4. Optional: Train neural network for faster evaluation

## License

MIT License
