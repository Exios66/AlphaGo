import pygame
import sys
from typing import Tuple, Optional
import torch
import numpy as np

from src.game.board import Board
from src.ai.policy_network import PolicyNetwork
from src.ai.value_network import ValueNetwork
from src.ai.mcts import MCTS

class GameUI:
    """
    A Pygame-based UI for the Go game.
    """
    
    def __init__(self, board_size: int = 19, ai_opponent: bool = False):
        """
        Initialize the game UI.
        
        Args:
            board_size (int): Size of the Go board (default: 19)
            ai_opponent (bool): Whether to play against AI
        """
        pygame.init()
        self.board = Board(board_size)
        self.ai_opponent = ai_opponent
        
        if ai_opponent:
            # Initialize AI components
            self.policy_net = PolicyNetwork(board_size)
            self.value_net = ValueNetwork(board_size)
            self.mcts = MCTS(self.policy_net, self.value_net)
            
            # Load pre-trained models if available
            try:
                self.policy_net.load_state_dict(torch.load('data/models/policy_net.pth'))
                self.value_net.load_state_dict(torch.load('data/models/value_net.pth'))
                print("Loaded pre-trained models")
            except:
                print("No pre-trained models found, using untrained networks")
        
        # UI constants
        self.CELL_SIZE = 40
        self.MARGIN = 50
        self.DOT_RADIUS = 4
        self.STONE_RADIUS = int(self.CELL_SIZE * 0.45)
        
        # Calculate window size
        board_pixels = board_size * self.CELL_SIZE
        self.window_size = (board_pixels + 2 * self.MARGIN, board_pixels + 2 * self.MARGIN)
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("AlphaGo Clone")
        
        # Colors
        self.BACKGROUND = (219, 179, 119)  # Wooden color
        self.LINE_COLOR = (0, 0, 0)
        self.DOT_COLOR = (0, 0, 0)
        self.BLACK_STONE = (0, 0, 0)
        self.WHITE_STONE = (255, 255, 255)
        self.TEXT_COLOR = (0, 0, 0)
        
        # Star points (for 19x19 board)
        self.star_points = []
        if board_size == 19:
            self.star_points = [
                (3, 3), (3, 9), (3, 15),
                (9, 3), (9, 9), (9, 15),
                (15, 3), (15, 9), (15, 15)
            ]
        
        # Game state
        self.game_over = False
        self.message = ""
        
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
        
    def _make_ai_move(self):
        """Make a move using the AI opponent."""
        if not self.game_over and self.ai_opponent and self.board.current_player == 2:  # AI plays as White
            # Get valid moves
            valid_moves = np.zeros((self.board.size, self.board.size), dtype=np.float32)
            legal_moves = self.board.get_legal_moves()
            for x, y in legal_moves:
                valid_moves[x, y] = 1
                
            # Select move using MCTS
            move = self.mcts.select_move(self.board)
            
            # Make the move
            if self.board.place_stone(*move):
                print(f"AI placed stone at {move}")
                black_score, white_score = self.board.get_score()
                print(f"Score - Black: {black_score}, White: {white_score}")
                
                # Check if game is over
                if len(self.board.get_legal_moves()) == 0:
                    self.game_over = True
                    self.message = f"Game Over! Black: {black_score}, White: {white_score}"
        
    def draw_board(self):
        """Draw the game board."""
        # Fill background
        self.screen.fill(self.BACKGROUND)
        
        # Draw grid lines
        for i in range(self.board.size):
            start_x, start_y = self._board_coords_to_pixels(i, 0)
            end_x, end_y = self._board_coords_to_pixels(i, self.board.size - 1)
            pygame.draw.line(self.screen, self.LINE_COLOR, (start_x, start_y), (start_x, end_y))
            
            start_x, start_y = self._board_coords_to_pixels(0, i)
            end_x, end_y = self._board_coords_to_pixels(self.board.size - 1, i)
            pygame.draw.line(self.screen, self.LINE_COLOR, (start_x, start_y), (end_x, end_y))
            
        # Draw star points
        for x, y in self.star_points:
            px, py = self._board_coords_to_pixels(x, y)
            pygame.draw.circle(self.screen, self.DOT_COLOR, (px, py), self.DOT_RADIUS)
            
        # Draw stones
        for x in range(self.board.size):
            for y in range(self.board.size):
                if self.board.board[x, y] != 0:
                    px, py = self._board_coords_to_pixels(x, y)
                    color = self.BLACK_STONE if self.board.board[x, y] == 1 else self.WHITE_STONE
                    pygame.draw.circle(self.screen, color, (px, py), self.STONE_RADIUS)
                    if color == self.WHITE_STONE:
                        pygame.draw.circle(self.screen, self.LINE_COLOR, (px, py), self.STONE_RADIUS, 1)
                        
        # Draw coordinates
        font = pygame.font.Font(None, 24)
        for i in range(self.board.size):
            # Draw column coordinates (A-T, excluding I)
            label = chr(ord('A') + i + (1 if i >= 8 else 0))
            text = font.render(label, True, self.LINE_COLOR)
            x, _ = self._board_coords_to_pixels(i, 0)
            self.screen.blit(text, (x - text.get_width()//2, 10))
            
            # Draw row coordinates (1-19)
            label = str(i + 1)
            text = font.render(label, True, self.LINE_COLOR)
            _, y = self._board_coords_to_pixels(0, i)
            self.screen.blit(text, (10, y - text.get_height()//2))
            
        # Draw current player and score
        black_score, white_score = self.board.get_score()
        current = "Black" if self.board.current_player == 1 else "White"
        status_text = f"Current Player: {current} | Black: {black_score:.1f} | White: {white_score:.1f}"
        if self.message:
            status_text = self.message
            
        text = font.render(status_text, True, self.TEXT_COLOR)
        self.screen.blit(text, (self.MARGIN, self.window_size[1] - 30))
            
        pygame.display.flip()
        
    def run(self):
        """Run the game loop."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and not self.game_over:
                    if event.button == 1:  # Left click
                        # Only allow human moves when it's their turn
                        if not self.ai_opponent or self.board.current_player == 1:
                            mouse_pos = pygame.mouse.get_pos()
                            board_pos = self._pixels_to_board_coords(*mouse_pos)
                            if board_pos:
                                if self.board.place_stone(*board_pos):
                                    print(f"Placed stone at {board_pos}")
                                    black_score, white_score = self.board.get_score()
                                    print(f"Score - Black: {black_score}, White: {white_score}")
                                    
                                    # Check if game is over
                                    if len(self.board.get_legal_moves()) == 0:
                                        self.game_over = True
                                        self.message = f"Game Over! Black: {black_score}, White: {white_score}"
                                        
            # Make AI move if it's AI's turn
            self._make_ai_move()
            
            self.draw_board()
            
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    ui = GameUI()
    ui.run() 