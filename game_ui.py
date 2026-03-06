"""
Pygame UI for 2048 — supports both the Expectimax AI and the trained
neural network player.

At startup the UI checks for model_checkpoint.pth:
  - Found  → NeuralPlayer (CNN trained via PPO)
  - Missing → AIPlayer    (Expectimax fallback)

After each game the checkpoint is automatically reloaded if it has
been updated on disk, so the player improves visibly while training
runs in the background.

Controls
--------
SPACE  pause / resume
R      reload checkpoint & start new game
Q      quit
"""

import pygame
import sys
import time
import os
from game2048 import Game2048
from ai_player import AIPlayer

# Optional neural player (requires torch + model_checkpoint.pth)
try:
    from neural_player import NeuralPlayer, CHECKPOINT_PATH
    _NEURAL_AVAILABLE = True
except ImportError:
    _NEURAL_AVAILABLE = False

# Optional n-tuple player (requires ntuple.bin produced by td_trainer.py)
try:
    from ntuple_player import NTuplePlayer, WEIGHTS_PATH as _NTUPLE_WEIGHTS_PATH
    _NTUPLE_AVAILABLE = True
except ImportError:
    _NTUPLE_AVAILABLE = False

# ── Colors ────────────────────────────────────────────────────────────────────
BACKGROUND = (187, 173, 160)
TILE_COLORS = {
    0:     (205, 193, 180),
    2:     (238, 228, 218),
    4:     (237, 224, 200),
    8:     (242, 177, 121),
    16:    (245, 149,  99),
    32:    (246, 124,  95),
    64:    (246,  94,  59),
    128:   (237, 207, 114),
    256:   (237, 204,  97),
    512:   (237, 200,  80),
    1024:  (237, 197,  63),
    2048:  (237, 194,  46),
    4096:  ( 60,  58,  50),
    8192:  ( 30,  30,  50),
    16384: ( 10,  10,  40),
    32768: (  5,   5,  30),
}

DARK_TEXT  = (119, 110, 101)
LIGHT_TEXT = (249, 246, 242)


# ── Helper ────────────────────────────────────────────────────────────────────
def _create_ai(neural_player: 'NeuralPlayer | None' = None,
               ntuple_player: 'NTuplePlayer | None' = None):
    """Return (ai_instance, label_string).

    Priority: N-Tuple AI > Neural AI > Expectimax AI.
    """
    # N-Tuple player (preferred when weights exist)
    if _NTUPLE_AVAILABLE and ntuple_player is not None and ntuple_player.loaded:
        return ntuple_player, "N-Tuple AI"
    if _NTUPLE_AVAILABLE and NTuplePlayer.is_available():
        nt = ntuple_player if ntuple_player is not None else NTuplePlayer()
        if nt.loaded:
            return nt, "N-Tuple AI"

    # Neural player fallback
    if _NEURAL_AVAILABLE and neural_player is not None and neural_player.loaded:
        return neural_player, "Neural AI"
    if _NEURAL_AVAILABLE and NeuralPlayer.is_available():
        np_instance = neural_player if neural_player is not None else NeuralPlayer()
        if np_instance.loaded:
            return np_instance, "Neural AI"

    return AIPlayer(search_depth=4), "Expectimax AI"


# ── UI class ──────────────────────────────────────────────────────────────────
class GameUI:
    """Pygame UI for watching the AI play 2048."""

    def __init__(self, cell_size: int = 100, padding: int = 10):
        pygame.init()

        self.cell_size   = cell_size
        self.padding     = padding
        self.board_size  = 4

        self.grid_size    = self.board_size * cell_size + (self.board_size + 1) * padding
        self.header_height = 130
        self.window_size  = (self.grid_size, self.grid_size + self.header_height)

        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("2048 AI Player")

        self.title_font      = pygame.font.Font(None, 60)
        self.score_font      = pygame.font.Font(None, 36)
        self.tile_font_large  = pygame.font.Font(None, 55)
        self.tile_font_medium = pygame.font.Font(None, 45)
        self.tile_font_small  = pygame.font.Font(None, 36)
        self.info_font       = pygame.font.Font(None, 26)

        self.animation_speed = 0.10  # seconds between moves
        self.paused          = False
        self.clock           = pygame.time.Clock()

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw_board(self, game: Game2048, ai_label: str = "",
                   ai_thinking: bool = False, reload_msg: str = ""):
        self.screen.fill(BACKGROUND)
        self._draw_header(game, ai_label, ai_thinking, reload_msg)
        for i in range(self.board_size):
            for j in range(self.board_size):
                self._draw_tile(i, j, game.board[i, j])
        pygame.display.flip()

    def _draw_header(self, game: Game2048, ai_label: str,
                     ai_thinking: bool, reload_msg: str):
        # Title
        title = self.title_font.render("2048 AI", True, DARK_TEXT)
        self.screen.blit(title, (20, 15))

        # Score
        score_surf = self.score_font.render(f"Score: {game.score}", True, DARK_TEXT)
        self.screen.blit(score_surf, (20, 70))

        # Max tile
        max_surf = self.score_font.render(f"Max: {game.get_max_tile()}", True, DARK_TEXT)
        self.screen.blit(max_surf, (200, 70))

        # AI type label (top-right)
        if "N-Tuple" in ai_label:
            label_color = (160, 80, 0)
        elif "Neural" in ai_label:
            label_color = (30, 120, 30)
        else:
            label_color = (80, 80, 160)
        label_surf = self.info_font.render(ai_label, True, label_color)
        self.screen.blit(label_surf, (self.grid_size - label_surf.get_width() - 15, 15))

        # Status line (thinking / paused / reload message)
        if reload_msg:
            status = reload_msg
            status_color = (180, 80, 0)
        elif ai_thinking:
            status = "Thinking…"
            status_color = DARK_TEXT
        elif self.paused:
            status = "PAUSED"
            status_color = (200, 0, 0)
        else:
            status = "SPACE: pause   R: new game   Q: quit"
            status_color = DARK_TEXT

        status_surf = self.info_font.render(status, True, status_color)
        self.screen.blit(status_surf,
                         (self.grid_size - status_surf.get_width() - 15, 45))

        # Neural training info (bottom of header, small text)
        if "Neural" in ai_label:
            info_surf = self.info_font.render(
                "SPACE: pause   R: new game / reload   Q: quit",
                True, DARK_TEXT
            )
            self.screen.blit(info_surf, (20, 105))

    def _draw_tile(self, row: int, col: int, value: int):
        x = col * (self.cell_size + self.padding) + self.padding
        y = row * (self.cell_size + self.padding) + self.padding + self.header_height

        color = TILE_COLORS.get(value, TILE_COLORS[32768])
        rect  = pygame.Rect(x, y, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, color, rect, border_radius=8)

        if value > 0:
            font = (
                self.tile_font_large  if value < 100   else
                self.tile_font_medium if value < 1000  else
                self.tile_font_small
            )
            text_color = LIGHT_TEXT if value > 4 else DARK_TEXT
            text = font.render(str(value), True, text_color)
            text_rect = text.get_rect(
                center=(x + self.cell_size // 2, y + self.cell_size // 2)
            )
            self.screen.blit(text, text_rect)

    def show_game_over(self, game: Game2048, ai_label: str):
        overlay = pygame.Surface(self.window_size)
        overlay.set_alpha(200)
        overlay.fill((255, 255, 255))
        self.screen.blit(overlay, (0, 0))

        cx = self.grid_size // 2
        cy = self.window_size[1] // 2

        go_font = pygame.font.Font(None, 80)
        go_surf  = go_font.render("Game Over!", True, (200, 0, 0))
        self.screen.blit(go_surf, go_surf.get_rect(center=(cx, cy - 110)))

        score_surf = self.score_font.render(f"Score: {game.score}", True, DARK_TEXT)
        self.screen.blit(score_surf, score_surf.get_rect(center=(cx, cy - 40)))

        tile_surf = self.score_font.render(
            f"Max tile: {game.get_max_tile()}", True, DARK_TEXT
        )
        self.screen.blit(tile_surf, tile_surf.get_rect(center=(cx, cy + 10)))

        ai_surf = self.info_font.render(f"AI: {ai_label}", True, DARK_TEXT)
        self.screen.blit(ai_surf, ai_surf.get_rect(center=(cx, cy + 55)))

        hint_surf = self.info_font.render(
            "R — new game / reload checkpoint    Q — quit", True, DARK_TEXT
        )
        self.screen.blit(hint_surf, hint_surf.get_rect(center=(cx, cy + 95)))

        pygame.display.flip()

    # ── Event handling ────────────────────────────────────────────────────────

    def handle_events(self):
        """Returns (continue_running, restart_requested)."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return False, False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    return True, True   # restart / reload
        return True, False


# ── Main entry point ──────────────────────────────────────────────────────────

def play_with_ui(search_depth: int = 4, speed_multiplier: float = 1.0):
    """Play 2048 with the UI.  Uses NeuralPlayer when a checkpoint exists,
    falls back to Expectimax AIPlayer otherwise."""

    ui = GameUI()
    ui.animation_speed /= speed_multiplier

    # Initialise players once (reused across games)
    ntuple_player = None
    if _NTUPLE_AVAILABLE and NTuplePlayer.is_available():
        print("Loading N-Tuple weights…")
        ntuple_player = NTuplePlayer()

    neural_player = None
    if _NEURAL_AVAILABLE:
        neural_player = NeuralPlayer()

    running = True
    while running:
        # ── Choose AI for this game ───────────────────────────────────────
        ai, ai_label = _create_ai(neural_player, ntuple_player)

        game = Game2048()
        moves_made = 0
        reload_msg = ""
        reload_msg_until = 0.0

        # ── Game loop ─────────────────────────────────────────────────────
        game_running = True
        while game_running and running and not game.game_over:
            running, restart = ui.handle_events()
            if not running:
                break
            if restart:
                # Reload checkpoint and start a fresh game immediately
                if neural_player is not None:
                    updated = neural_player.reload_if_updated()
                    ai, ai_label = _create_ai(neural_player)
                    reload_msg = (
                        f"Checkpoint reloaded!  ep:{neural_player.episodes_done:,}"
                        if updated else "No new checkpoint yet."
                    )
                    reload_msg_until = time.time() + 2.5
                    ai, ai_label = _create_ai(neural_player, ntuple_player)
                game_running = False
                break

            if ui.paused:
                ui.draw_board(game, ai_label, ai_thinking=False)
                ui.clock.tick(30)
                continue

            # Expire temporary reload message
            active_msg = reload_msg if time.time() < reload_msg_until else ""

            # Show thinking state
            ui.draw_board(game, ai_label, ai_thinking=True, reload_msg=active_msg)
            pygame.event.pump()

            move = ai.get_best_move(game)
            if move is None:
                break

            if game.move(move):
                moves_made += 1
                ui.draw_board(game, ai_label, ai_thinking=False, reload_msg=active_msg)
                time.sleep(ui.animation_speed)

        if not running:
            break

        if game.game_over:
            # Show game-over screen
            ui.draw_board(game, ai_label, ai_thinking=False)
            time.sleep(0.4)
            ui.show_game_over(game, ai_label)

            # Reload checkpoint while waiting for user input
            if neural_player is not None:
                neural_player.reload_if_updated()

            print(f"\nGame finished — {ai_label}")
            print(f"  Score : {game.score:,}")
            print(f"  Max   : {game.get_max_tile()}")
            print(f"  Moves : {moves_made}")
            if hasattr(neural_player, 'episodes_done') and neural_player is not None:
                print(f"  Model trained for {neural_player.episodes_done:,} episodes")

            # Wait for R (new game) or Q (quit)
            waiting = True
            while waiting and running:
                running, restart = ui.handle_events()
                if restart or not running:
                    waiting = False
                ui.clock.tick(30)

            # Always loop back: a new game will re-read AI after the break
            # (running=False exits the outer while)

    pygame.quit()


if __name__ == "__main__":
    play_with_ui(search_depth=4, speed_multiplier=1.5)
