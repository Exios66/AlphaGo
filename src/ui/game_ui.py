import pygame
import sys
import os
import logging
from typing import Tuple, Optional, Dict
import torch
import numpy as np
from pathlib import Path

from src.game.board import Board
from src.ai.policy_network import PolicyNetwork 
from src.ai.value_network import ValueNetwork
from src.ai.mcts import MCTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('game.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GameUI:
    """
    A production-ready Pygame-based UI for the Go game.
    
    Features:
    - Configurable board sizes (9x9, 13x13, 19x19)
    - AI opponent with configurable strength
    - Game state saving/loading
    - Move history and replay
    - Score tracking
    - Sound effects
    - Error handling and logging
    """
    
    # Class constants
    VALID_BOARD_SIZES = [9, 13, 19]
    DEFAULT_BOARD_SIZE = 19
    MODEL_DIR = Path('data/models')
    SOUND_DIR = Path('data/sounds')
    SAVE_DIR = Path('data/saves')
    
    def __init__(self, 
                 board_size: int = DEFAULT_BOARD_SIZE, 
                 ai_opponent: bool = False,
                 num_simulations: int = 800,
                 sound_enabled: bool = True):
        """
        Initialize the game UI.
        
        Args:
            board_size: Size of the Go board (9, 13, or 19)
            ai_opponent: Whether to play against AI
            num_simulations: Number of MCTS simulations per move
            sound_enabled: Whether to play sound effects
        """
        # Validate board size
        if board_size not in self.VALID_BOARD_SIZES:
            logger.warning(f"Invalid board size {board_size}. Using default size {self.DEFAULT_BOARD_SIZE}")
            board_size = self.DEFAULT_BOARD_SIZE
            
        # Initialize pygame and mixer
        try:
            pygame.init()
            if sound_enabled:
                pygame.mixer.init()
        except pygame.error as e:
            logger.error(f"Failed to initialize pygame: {e}")
            raise
            
        # Create required directories
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.SOUND_DIR.mkdir(parents=True, exist_ok=True) 
        self.SAVE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Game state
        self.board = Board(board_size)
        self.ai_opponent = ai_opponent
        self.sound_enabled = sound_enabled
        self.game_over = False
        self.message = ""
        self.move_history = []
        self.current_move_idx = -1
        
        # Load AI components if needed
        if ai_opponent:
            self._initialize_ai(num_simulations)
            
        # Load sound effects
        self.sounds: Dict[str, Optional[pygame.mixer.Sound]] = {}
        if sound_enabled:
            self._load_sound_effects()
            
        # UI constants
        self.CELL_SIZE = 40
        self.MARGIN = 50
        self.DOT_RADIUS = 4
        self.STONE_RADIUS = int(self.CELL_SIZE * 0.45)
        
        # Colors
        self.COLORS = {
            'background': (219, 179, 119),
            'line': (0, 0, 0),
            'dot': (0, 0, 0),
            'black_stone': (0, 0, 0),
            'white_stone': (255, 255, 255),
            'text': (0, 0, 0),
            'highlight': (255, 0, 0, 128)
        }
        
        # Calculate window size and initialize display
        board_pixels = board_size * self.CELL_SIZE
        self.window_size = (board_pixels + 2 * self.MARGIN, board_pixels + 2 * self.MARGIN)
        
        try:
            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("AlphaGo Clone")
        except pygame.error as e:
            logger.error(f"Failed to create display: {e}")
            raise
            
        # Initialize star points
        self.star_points = self._get_star_points(board_size)
        
        logger.info(f"Initialized game UI with {board_size}x{board_size} board")
        
    def _initialize_ai(self, num_simulations: int):
        """Initialize and load AI components."""
        try:
            self.policy_net = PolicyNetwork(self.board.size)
            self.value_net = ValueNetwork(self.board.size)
            self.mcts = MCTS(self.policy_net, self.value_net, num_simulations=num_simulations)
            
            # Load pre-trained models
            policy_path = self.MODEL_DIR / 'policy_net.pth'
            value_path = self.MODEL_DIR / 'value_net.pth'
            
            if policy_path.exists() and value_path.exists():
                self.policy_net.load_state_dict(torch.load(policy_path))
                self.value_net.load_state_dict(torch.load(value_path))
                logger.info("Loaded pre-trained models")
            else:
                logger.warning("No pre-trained models found, using untrained networks")
                
        except Exception as e:
            logger.error(f"Failed to initialize AI: {e}")
            raise
            
    def _load_sound_effects(self):
        """Load game sound effects."""
        sound_files = {
            'stone': 'stone.wav',
            'capture': 'capture.wav',
            'illegal': 'illegal.wav',
            'game_over': 'game_over.wav'
        }
        
        for name, file in sound_files.items():
            try:
                path = self.SOUND_DIR / file
                if path.exists():
                    self.sounds[name] = pygame.mixer.Sound(str(path))
                else:
                    logger.warning(f"Sound file not found: {path}")
                    self.sounds[name] = None
            except pygame.error as e:
                logger.error(f"Failed to load sound {file}: {e}")
                self.sounds[name] = None
                
    def _get_star_points(self, board_size: int) -> list:
        """Get star point positions for different board sizes."""
        if board_size == 19:
            return [(3, 3), (3, 9), (3, 15),
                    (9, 3), (9, 9), (9, 15),
                    (15, 3), (15, 9), (15, 15)]
        elif board_size == 13:
            return [(3, 3), (3, 9),
                    (6, 6),
                    (9, 3), (9, 9)]
        elif board_size == 9:
            return [(2, 2), (2, 6),
                    (4, 4),
                    (6, 2), (6, 6)]
        return []
        
    def _board_coords_to_pixels(self, x: int, y: int) -> Tuple[int, int]:
        """Convert board coordinates to pixel coordinates."""
        return (
            x * self.CELL_SIZE + self.MARGIN,
            y * self.CELL_SIZE + self.MARGIN
        )
        
    def _pixels_to_board_coords(self, px: int, py: int) -> Optional[Tuple[int, int]]:
        """Convert pixel coordinates to board coordinates."""
        x = (px - self.MARGIN + self.CELL_SIZE // 2) // self.CELL_SIZE
        y = (py - self.MARGIN + self.CELL_SIZE // 2) // self.CELL_SIZE
        
        if 0 <= x < self.board.size and 0 <= y < self.board.size:
            return (x, y)
        return None
        
    def _play_sound(self, sound_name: str):
        """Play a sound effect if enabled."""
        if self.sound_enabled and sound_name in self.sounds and self.sounds[sound_name]:
            try:
                self.sounds[sound_name].play()
            except pygame.error as e:
                logger.error(f"Failed to play sound {sound_name}: {e}")
                
    def _make_ai_move(self):
        """Make a move using the AI opponent."""
        if not self.game_over and self.ai_opponent and self.board.current_player == 2:
            try:
                # Get valid moves
                valid_moves = np.zeros((self.board.size, self.board.size), dtype=np.float32)
                legal_moves = self.board.get_legal_moves()
                for x, y in legal_moves:
                    valid_moves[x, y] = 1
                    
                # Select and make move
                move = self.mcts.select_move(self.board)
                if self.board.place_stone(*move):
                    logger.info(f"AI placed stone at {move}")
                    self._play_sound('stone')
                    self.move_history.append(move)
                    self.current_move_idx += 1
                    
                    # Update game state
                    black_score, white_score = self.board.get_score()
                    logger.info(f"Score - Black: {black_score}, White: {white_score}")
                    
                    if len(self.board.get_legal_moves()) == 0:
                        self._end_game()
                        
            except Exception as e:
                logger.error(f"Error during AI move: {e}")
                
    def _end_game(self):
        """Handle game end conditions."""
        self.game_over = True
        black_score, white_score = self.board.get_score()
        self.message = f"Game Over! Black: {black_score:.1f}, White: {white_score:.1f}"
        self._play_sound('game_over')
        logger.info(f"Game ended - {self.message}")
        
    def save_game(self, filename: str):
        """Save current game state."""
        try:
            save_path = self.SAVE_DIR / f"{filename}.npz"
            np.savez(save_path,
                    board=self.board.board,
                    current_player=self.board.current_player,
                    move_history=self.move_history,
                    captured_stones=self.board.captured_stones)
            logger.info(f"Game saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save game: {e}")
            
    def load_game(self, filename: str) -> bool:
        """Load a saved game state."""
        try:
            save_path = self.SAVE_DIR / f"{filename}.npz"
            if not save_path.exists():
                logger.error(f"Save file not found: {save_path}")
                return False
                
            data = np.load(save_path)
            self.board.board = data['board']
            self.board.current_player = int(data['current_player'])
            self.move_history = data['move_history'].tolist()
            self.board.captured_stones = dict(data['captured_stones'].item())
            self.current_move_idx = len(self.move_history) - 1
            
            logger.info(f"Game loaded from {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load game: {e}")
            return False
            
    def draw_board(self):
        """Draw the game board and UI elements."""
        try:
            # Fill background
            self.screen.fill(self.COLORS['background'])
            
            # Draw grid lines
            for i in range(self.board.size):
                start_x, start_y = self._board_coords_to_pixels(i, 0)
                end_x, end_y = self._board_coords_to_pixels(i, self.board.size - 1)
                pygame.draw.line(self.screen, self.COLORS['line'], 
                               (start_x, start_y), (start_x, end_y))
                
                start_x, start_y = self._board_coords_to_pixels(0, i)
                end_x, end_y = self._board_coords_to_pixels(self.board.size - 1, i)
                pygame.draw.line(self.screen, self.COLORS['line'],
                               (start_x, start_y), (end_x, end_y))
                
            # Draw star points
            for x, y in self.star_points:
                px, py = self._board_coords_to_pixels(x, y)
                pygame.draw.circle(self.screen, self.COLORS['dot'], 
                                 (px, py), self.DOT_RADIUS)
                
            # Draw stones
            for x in range(self.board.size):
                for y in range(self.board.size):
                    if self.board.board[x, y] != 0:
                        px, py = self._board_coords_to_pixels(x, y)
                        color = self.COLORS['black_stone'] if self.board.board[x, y] == 1 \
                               else self.COLORS['white_stone']
                        pygame.draw.circle(self.screen, color, (px, py), self.STONE_RADIUS)
                        if color == self.COLORS['white_stone']:
                            pygame.draw.circle(self.screen, self.COLORS['line'],
                                            (px, py), self.STONE_RADIUS, 1)
                            
            # Draw coordinates and status
            self._draw_coordinates()
            self._draw_status()
            
            pygame.display.flip()
            
        except pygame.error as e:
            logger.error(f"Error drawing board: {e}")
            
    def _draw_coordinates(self):
        """Draw board coordinates."""
        font = pygame.font.Font(None, 24)
        for i in range(self.board.size):
            # Column coordinates (A-T, excluding I)
            label = chr(ord('A') + i + (1 if i >= 8 else 0))
            text = font.render(label, True, self.COLORS['text'])
            x, _ = self._board_coords_to_pixels(i, 0)
            self.screen.blit(text, (x - text.get_width()//2, 10))
            
            # Row coordinates (1-19)
            label = str(i + 1)
            text = font.render(label, True, self.COLORS['text'])
            _, y = self._board_coords_to_pixels(0, i)
            self.screen.blit(text, (10, y - text.get_height()//2))
            
    def _draw_status(self):
        """Draw game status information."""
        font = pygame.font.Font(None, 24)
        black_score, white_score = self.board.get_score()
        current = "Black" if self.board.current_player == 1 else "White"
        
        status_text = f"Current Player: {current} | "
        status_text += f"Black: {black_score:.1f} | White: {white_score:.1f} | "
        status_text += f"Captures - Black: {self.board.captured_stones[1]}, "
        status_text += f"White: {self.board.captured_stones[2]}"
        
        if self.message:
            status_text = self.message
            
        text = font.render(status_text, True, self.COLORS['text'])
        self.screen.blit(text, (self.MARGIN, self.window_size[1] - 30))
        
    def run(self):
        """Run the game loop."""
        running = True
        clock = pygame.time.Clock()
        
        try:
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.MOUSEBUTTONDOWN and not self.game_over:
                        self._handle_mouse_click(event)
                    elif event.type == pygame.KEYDOWN:
                        self._handle_keyboard_input(event)
                        
                # Make AI move if needed
                self._make_ai_move()
                
                # Update display
                self.draw_board()
                clock.tick(60)
                
        except Exception as e:
            logger.error(f"Error in game loop: {e}")
        finally:
            self._cleanup()
            
    def _handle_mouse_click(self, event):
        """Handle mouse click events."""
        if event.button == 1:  # Left click
            if not self.ai_opponent or self.board.current_player == 1:
                mouse_pos = pygame.mouse.get_pos()
                board_pos = self._pixels_to_board_coords(*mouse_pos)
                
                if board_pos:
                    if self.board.place_stone(*board_pos):
                        logger.info(f"Placed stone at {board_pos}")
                        self._play_sound('stone')
                        self.move_history.append(board_pos)
                        self.current_move_idx += 1
                        
                        if len(self.board.get_legal_moves()) == 0:
                            self._end_game()
                    else:
                        self._play_sound('illegal')
                        
    def _handle_keyboard_input(self, event):
        """Handle keyboard input events."""
        if event.key == pygame.K_s and event.mod & pygame.KMOD_CTRL:
            self.save_game('quicksave')
        elif event.key == pygame.K_l and event.mod & pygame.KMOD_CTRL:
            self.load_game('quicksave')
        elif event.key == pygame.K_z and event.mod & pygame.KMOD_CTRL:
            self._undo_move()
        elif event.key == pygame.K_r and event.mod & pygame.KMOD_CTRL:
            self._redo_move()
            
    def _undo_move(self):
        """Undo the last move."""
        if self.current_move_idx >= 0:
            self.board = Board(self.board.size)
            for i in range(self.current_move_idx):
                self.board.place_stone(*self.move_history[i])
            self.current_move_idx -= 1
            
    def _redo_move(self):
        """Redo a previously undone move."""
        if self.current_move_idx < len(self.move_history) - 1:
            self.current_move_idx += 1
            self.board.place_stone(*self.move_history[self.current_move_idx])
            
    def _cleanup(self):
        """Clean up resources before exit."""
        try:
            pygame.quit()
        except pygame.error as e:
            logger.error(f"Error during cleanup: {e}")
        sys.exit()

if __name__ == "__main__":
    try:
        ui = GameUI()
        ui.run()
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)