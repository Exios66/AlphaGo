from typing import List, Set, Tuple, Optional
import numpy as np

class Board:
    """
    Represents a Go board and implements the game rules.
    """
    
    def __init__(self, size: int = 19):
        """
        Initialize a Go board with the specified size.
        
        Args:
            size (int): Size of the board (default: 19)
        """
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: empty, 1: black, 2: white
        self.current_player = 1  # Black starts
        self.ko_point: Optional[Tuple[int, int]] = None
        self.move_history: List[Tuple[int, int]] = []
        self.captured_stones = {1: 0, 2: 0}  # Stones captured by each player
        
    def is_valid_move(self, x: int, y: int) -> bool:
        """
        Check if a move is valid at the specified position.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            
        Returns:
            bool: True if the move is valid, False otherwise
        """
        # Check if position is within bounds
        if not (0 <= x < self.size and 0 <= y < self.size):
            return False
            
        # Check if position is empty
        if self.board[x, y] != 0:
            return False
            
        # Check ko rule
        if self.ko_point == (x, y):
            return False
            
        # Place stone temporarily to check for suicide rule
        self.board[x, y] = self.current_player
        has_liberties = self._has_liberties(x, y)
        captures_enemy = self._check_captures(x, y)
        self.board[x, y] = 0
        
        return has_liberties or captures_enemy
        
    def place_stone(self, x: int, y: int) -> bool:
        """
        Place a stone at the specified position.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            
        Returns:
            bool: True if the move was successful, False otherwise
        """
        if not self.is_valid_move(x, y):
            return False
            
        self.board[x, y] = self.current_player
        captured = self._remove_captured_stones(x, y)
        self.captured_stones[self.current_player] += captured
        
        # Update ko point
        self.ko_point = None
        if captured == 1:
            self.ko_point = (x, y)
            
        self.move_history.append((x, y))
        self._switch_player()
        return True
        
    def _has_liberties(self, x: int, y: int) -> bool:
        """Check if a stone or group has liberties."""
        visited = set()
        return self._check_liberties(x, y, visited)
        
    def _check_liberties(self, x: int, y: int, visited: Set[Tuple[int, int]]) -> bool:
        """Recursive helper function to check for liberties."""
        if (x, y) in visited:
            return False
            
        visited.add((x, y))
        color = self.board[x, y]
        
        # Check adjacent positions
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.size and 0 <= ny < self.size):
                continue
                
            if self.board[nx, ny] == 0:
                return True
            if self.board[nx, ny] == color and self._check_liberties(nx, ny, visited):
                return True
                
        return False
        
    def _check_captures(self, x: int, y: int) -> bool:
        """Check if placing a stone at (x,y) captures any enemy stones."""
        opponent = 3 - self.current_player
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.size and 0 <= ny < self.size):
                continue
            if self.board[nx, ny] == opponent and not self._has_liberties(nx, ny):
                return True
        return False
        
    def _remove_captured_stones(self, x: int, y: int) -> int:
        """Remove captured stones after a move and return the count."""
        opponent = 3 - self.current_player
        captured = 0
        
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.size and 0 <= ny < self.size):
                continue
                
            if self.board[nx, ny] == opponent:
                group = self._get_group(nx, ny)
                if not any(self._has_liberties(gx, gy) for gx, gy in group):
                    captured += len(group)
                    for gx, gy in group:
                        self.board[gx, gy] = 0
                        
        return captured
        
    def _get_group(self, x: int, y: int) -> Set[Tuple[int, int]]:
        """Get all stones in the same group."""
        color = self.board[x, y]
        group = set()
        self._flood_fill(x, y, color, group)
        return group
        
    def _flood_fill(self, x: int, y: int, color: int, group: Set[Tuple[int, int]]):
        """Flood fill algorithm to find connected stones."""
        if not (0 <= x < self.size and 0 <= y < self.size):
            return
        if (x, y) in group or self.board[x, y] != color:
            return
            
        group.add((x, y))
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            self._flood_fill(x + dx, y + dy, color, group)
            
    def _switch_player(self):
        """Switch the current player."""
        self.current_player = 3 - self.current_player
        
    def get_legal_moves(self) -> List[Tuple[int, int]]:
        """Get all legal moves for the current player."""
        moves = []
        for x in range(self.size):
            for y in range(self.size):
                if self.is_valid_move(x, y):
                    moves.append((x, y))
        return moves
        
    def get_score(self) -> Tuple[float, float]:
        """
        Calculate the score using area scoring.
        Returns (black_score, white_score)
        """
        territory = np.zeros((self.size, self.size), dtype=int)
        visited = set()
        
        # Find territory
        for x in range(self.size):
            for y in range(self.size):
                if (x, y) not in visited and self.board[x, y] == 0:
                    territory_points = set()
                    color = self._find_territory_owner(x, y, territory_points)
                    if color > 0:
                        for tx, ty in territory_points:
                            territory[tx, ty] = color
                    visited.update(territory_points)
                    
        # Count territory and stones
        black_score = np.sum(territory == 1) + np.sum(self.board == 1)
        white_score = np.sum(territory == 2) + np.sum(self.board == 2)
        
        return black_score, white_score
        
    def _find_territory_owner(self, x: int, y: int, territory: Set[Tuple[int, int]]) -> int:
        """
        Find the owner of an empty region.
        Returns 1 for black, 2 for white, 0 for neutral.
        """
        if (x, y) in territory:
            return 0
            
        if not (0 <= x < self.size and 0 <= y < self.size):
            return 0
            
        if self.board[x, y] > 0:
            return self.board[x, y]
            
        territory.add((x, y))
        owners = set()
        
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            owner = self._find_territory_owner(x + dx, y + dy, territory)
            if owner > 0:
                owners.add(owner)
                
        if len(owners) == 1:
            return owners.pop()
        return 0  # Neutral territory
        
    def __str__(self) -> str:
        """String representation of the board."""
        symbols = {0: '.', 1: '●', 2: '○'}
        rows = []
        for i in range(self.size):
            row = ' '.join(symbols[stone] for stone in self.board[i])
            rows.append(f"{i:2d} {row}")
        
        header = '   ' + ' '.join(chr(ord('A') + i) for i in range(self.size))
        return header + '\n' + '\n'.join(rows) 