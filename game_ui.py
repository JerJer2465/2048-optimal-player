"""
Pygame UI for 2048 game with AI player.
Watch the AI play in real-time with smooth animations.
"""

import pygame
import sys
import time
from game2048 import Game2048
from ai_player import AIPlayer

# Colors
BACKGROUND = (187, 173, 160)
EMPTY_CELL = (205, 193, 180)
TILE_COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
    4096: (60, 58, 50),
}

DARK_TEXT = (119, 110, 101)
LIGHT_TEXT = (249, 246, 242)

class GameUI:
    """Pygame UI for 2048 game."""
    
    def __init__(self, cell_size=100, padding=10):
        """Initialize UI.
        
        Args:
            cell_size: Size of each cell in pixels
            padding: Padding between cells
        """
        pygame.init()
        
        self.cell_size = cell_size
        self.padding = padding
        self.board_size = 4
        
        # Calculate window size
        self.grid_size = self.board_size * cell_size + (self.board_size + 1) * padding
        self.header_height = 120
        self.window_size = (self.grid_size, self.grid_size + self.header_height)
        
        # Create window
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("2048 AI Player")
        
        # Fonts
        self.title_font = pygame.font.Font(None, 60)
        self.score_font = pygame.font.Font(None, 36)
        self.tile_font_large = pygame.font.Font(None, 55)
        self.tile_font_medium = pygame.font.Font(None, 45)
        self.tile_font_small = pygame.font.Font(None, 36)
        self.info_font = pygame.font.Font(None, 28)
        
        # Animation
        self.animation_speed = 0.15  # seconds per move
        self.paused = False
        
        # Clock
        self.clock = pygame.time.Clock()
    
    def draw_board(self, game: Game2048, ai_thinking: bool = False):
        """Draw the game board.
        
        Args:
            game: Current game state
            ai_thinking: Whether AI is thinking (show indicator)
        """
        self.screen.fill(BACKGROUND)
        
        # Draw header
        self._draw_header(game, ai_thinking)
        
        # Draw grid
        for i in range(self.board_size):
            for j in range(self.board_size):
                value = game.board[i, j]
                self._draw_tile(i, j, value)
        
        pygame.display.flip()
    
    def _draw_header(self, game: Game2048, ai_thinking: bool):
        """Draw header with title, score, and controls."""
        # Title
        title = self.title_font.render("2048 AI", True, DARK_TEXT)
        self.screen.blit(title, (20, 20))
        
        # Score
        score_text = f"Score: {game.score}"
        score = self.score_font.render(score_text, True, DARK_TEXT)
        self.screen.blit(score, (20, 75))
        
        # Max tile
        max_tile = game.get_max_tile()
        max_text = f"Max: {max_tile}"
        max_render = self.score_font.render(max_text, True, DARK_TEXT)
        self.screen.blit(max_render, (200, 75))
        
        # Controls
        controls = "SPACE: Pause  Q: Quit"
        if ai_thinking:
            controls = "Thinking..."
        controls_render = self.info_font.render(controls, True, DARK_TEXT)
        controls_x = self.grid_size - controls_render.get_width() - 20
        self.screen.blit(controls_render, (controls_x, 25))
        
        # Paused indicator
        if self.paused:
            paused_text = self.score_font.render("PAUSED", True, (255, 0, 0))
            paused_x = self.grid_size - paused_text.get_width() - 20
            self.screen.blit(paused_text, (paused_x, 75))
    
    def _draw_tile(self, row: int, col: int, value: int):
        """Draw a single tile.
        
        Args:
            row: Row index
            col: Column index
            value: Tile value (0 for empty)
        """
        x = col * (self.cell_size + self.padding) + self.padding
        y = row * (self.cell_size + self.padding) + self.padding + self.header_height
        
        # Get color
        color = TILE_COLORS.get(value, TILE_COLORS[4096])
        
        # Draw rounded rectangle
        rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, color, rect, border_radius=8)
        
        # Draw value
        if value > 0:
            # Choose font size based on number of digits
            if value < 100:
                font = self.tile_font_large
            elif value < 1000:
                font = self.tile_font_medium
            else:
                font = self.tile_font_small
            
            # Choose text color
            text_color = LIGHT_TEXT if value > 4 else DARK_TEXT
            
            # Render text
            text = font.render(str(value), True, text_color)
            text_rect = text.get_rect(center=(x + self.cell_size // 2,
                                               y + self.cell_size // 2))
            self.screen.blit(text, text_rect)
    
    def show_game_over(self, game: Game2048):
        """Display game over screen.
        
        Args:
            game: Final game state
        """
        # Semi-transparent overlay
        overlay = pygame.Surface(self.window_size)
        overlay.set_alpha(200)
        overlay.fill((255, 255, 255))
        self.screen.blit(overlay, (0, 0))
        
        # Game Over text
        game_over_font = pygame.font.Font(None, 80)
        game_over = game_over_font.render("Game Over!", True, (255, 0, 0))
        game_over_rect = game_over.get_rect(center=(self.grid_size // 2,
                                                      self.window_size[1] // 2 - 80))
        self.screen.blit(game_over, game_over_rect)
        
        # Final stats
        score_text = f"Final Score: {game.score}"
        score = self.score_font.render(score_text, True, DARK_TEXT)
        score_rect = score.get_rect(center=(self.grid_size // 2,
                                              self.window_size[1] // 2))
        self.screen.blit(score, score_rect)
        
        max_text = f"Max Tile: {game.get_max_tile()}"
        max_tile = self.score_font.render(max_text, True, DARK_TEXT)
        max_rect = max_tile.get_rect(center=(self.grid_size // 2,
                                               self.window_size[1] // 2 + 50))
        self.screen.blit(max_tile, max_rect)
        
        # Instructions
        instructions = "Press Q to quit"
        inst = self.info_font.render(instructions, True, DARK_TEXT)
        inst_rect = inst.get_rect(center=(self.grid_size // 2,
                                           self.window_size[1] // 2 + 100))
        self.screen.blit(inst, inst_rect)
        
        pygame.display.flip()
    
    def handle_events(self):
        """Handle pygame events.
        
        Returns:
            True to continue, False to quit
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
        return True


def play_with_ui(search_depth: int = 4, speed_multiplier: float = 1.0):
    """Play game with UI.
    
    Args:
        search_depth: AI search depth (3-5 recommended)
        speed_multiplier: Speed multiplier (1.0 = normal, 2.0 = 2x fast)
    """
    game = Game2048()
    ai = AIPlayer(search_depth=search_depth)
    ui = GameUI()
    
    ui.animation_speed /= speed_multiplier
    
    moves_made = 0
    running = True
    
    while running and not game.game_over:
        # Handle events
        running = ui.handle_events()
        if not running:
            break
        
        # Draw current state
        if ui.paused:
            ui.draw_board(game, ai_thinking=False)
            ui.clock.tick(30)
            continue
        
        # Show thinking state
        ui.draw_board(game, ai_thinking=True)
        pygame.event.pump()  # Process events during thinking
        
        # Get AI move
        move = ai.get_best_move(game)
        
        if move is None:
            break
        
        # Execute move
        if game.move(move):
            moves_made += 1
            
            # Draw result with small delay
            ui.draw_board(game, ai_thinking=False)
            time.sleep(ui.animation_speed)
    
    # Show final state
    if running:
        ui.draw_board(game, ai_thinking=False)
        time.sleep(0.5)
        ui.show_game_over(game)
        
        # Wait for quit
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or \
                   (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    waiting = False
            ui.clock.tick(30)
    
    pygame.quit()
    
    print(f"\nGame finished!")
    print(f"Final score: {game.score}")
    print(f"Max tile: {game.get_max_tile()}")
    print(f"Total moves: {moves_made}")


if __name__ == "__main__":
    # Default: depth 4, normal speed
    # For faster: play_with_ui(search_depth=4, speed_multiplier=2.0)
    play_with_ui(search_depth=4, speed_multiplier=1.5)
