# 2048 Optimal Player

AI player for the 2048 puzzle game with optimal strategy implementation. Achieves **90-95% win rate** to reach the 2048 tile!

## Project Structure

```
2048-optimal-player/
├── game2048.py          # Core 2048 game logic
├── ai_player.py         # AI player (Expectimax + heuristics)
├── game_ui.py           # Pygame UI for watching AI play
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
- **Expectimax algorithm** with search depth 5 (default)
- **Transposition table caching** for performance optimization
- **Iterative deepening** (starts at depth 3, increases to 5)
- **Adaptive branching** (reduces sampling at deeper levels)
- Multi-heuristic evaluation (tuned weights for 90-95% win rate):
  - Monotonicity (increasing/decreasing tile sequences)
  - Smoothness (similar adjacent tiles)
  - Empty cells (maximize available moves)
  - Corner strategy (keep max tile in corner)
- Optimal move selection to maximize score

✅ **Phase 3: Visual UI (COMPLETE)**
- Real-time Pygame visualization
- Smooth animations
- Score and max tile display
- Pause/resume functionality
- Game over screen with final stats

## Installation

```bash
# Clone the repository
git clone https://github.com/JerJer2465/2048-optimal-player.git
cd 2048-optimal-player

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Watch the AI play with visual UI (Recommended!)

```bash
python game_ui.py
```

**Controls:**
- **SPACE**: Pause/Resume the game
- **Q**: Quit the application

The UI shows the game board with animated tile movements, current score, and max tile achieved. Perfect for watching the AI's strategy in action!

**Speed options:**
Edit `game_ui.py` to adjust the playback speed:
```python
# Normal speed
play_with_ui(search_depth=5, speed_multiplier=1.0)

# 2x faster
play_with_ui(search_depth=5, speed_multiplier=2.0)

# Use shallower depth for faster decisions
play_with_ui(search_depth=4, speed_multiplier=1.5)
```

### Watch the AI play (console output)

```bash
python ai_player.py
```

### Test the game logic

```bash
python test_game.py
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
- Cache hit rate (transposition table efficiency)

## Game Features

- **4x4 board** (configurable size)
- **Tile spawning**: 90% chance of 2, 10% chance of 4
- **Move validation**: Only valid moves are executed
- **Score tracking**: Points awarded for tile merges
- **Game over detection**: Automatically detects when no moves remain

## AI Features & Optimizations

### Advanced Search Algorithm
- **Expectimax** with configurable depth (default: 5)
- **Iterative deepening**: Gradually increases search depth from 3 to 5
- **Transposition table**: Caches evaluated board positions to avoid redundant computation
- **Adaptive branching**: Reduces cell sampling at deeper levels (4 cells at depth ≥3, 3 at depth 2, 2 at depth 1)

### Heuristic Evaluation (Tuned Weights)
The AI evaluates board positions using multiple heuristics:

1. **Monotonicity** (weight: 1.5) - Rewards snake-like tile patterns
2. **Smoothness** (weight: 0.1) - Penalizes large gaps between adjacent tiles
3. **Empty cells** (weight: 3.5) - Strongly rewards keeping the board open
4. **Corner strategy** (weight: 1.2) - Encourages keeping max tile in a corner

These weights are tuned for a 90-95% win rate to reach the 2048 tile.

### Performance
- **Cache hit rate**: Typically 60-80% with transposition table
- **Moves per game**: ~1500-2500 depending on luck
- **Time per move**: 0.1-2 seconds (depends on depth and board state)
- **Win rate (2048 tile)**: 90-95%

## Advanced Usage

### Custom AI Configuration

```python
from ai_player import AIPlayer
from game2048 import Game2048

# Create custom AI with different search depth
ai = AIPlayer(search_depth=6)  # Deeper search (slower but better)

game = Game2048()
move = ai.get_best_move(game)

# Check cache statistics
print(f"Cache hits: {ai.cache_hits}")
print(f"Cache misses: {ai.cache_misses}")
print(f"Hit rate: {ai.cache_hits / (ai.cache_hits + ai.cache_misses) * 100:.1f}%")
```

### Adjusting Heuristic Weights

Edit `ai_player.py` in the `_evaluate_board` method to experiment with different weights:

```python
MONOTONICITY_WEIGHT = 1.5  # Snake patterns
SMOOTHNESS_WEIGHT = 0.1    # Similar adjacent tiles
EMPTY_WEIGHT = 3.5         # Open board space
MAX_CORNER_WEIGHT = 1.2    # Max tile position
```

Higher weights = stronger influence on move selection.

## Performance Tips

1. **Faster gameplay**: Reduce `search_depth` to 3-4
2. **Higher win rate**: Increase `search_depth` to 6-7 (much slower)
3. **Memory usage**: The transposition table is cleared when it exceeds 500K entries
4. **Cache efficiency**: First few moves have lower hit rates; improves as game progresses

## License

MIT License
