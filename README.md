# 2048 Optimal Player

AI player for the 2048 puzzle game with optimal strategy implementation.

## Project Structure

```
2048-optimal-player/
├── game2048.py          # Core 2048 game logic
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

🔄 **Phase 2: AI Player (TODO)**
- Expectimax algorithm
- Heuristic evaluation function
- Optimal move selection

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
