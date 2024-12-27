import pytest
import numpy as np
from src.game.board import Board

@pytest.fixture
def empty_board():
    return Board(size=19)

@pytest.fixture 
def small_board():
    return Board(size=9)

class TestBoardInitialization:
    @pytest.mark.parametrize("size", [9, 13, 19])
    def test_board_creation(self, size):
        board = Board(size=size)
        assert board.size == size
        assert board.board.shape == (size, size)
        assert board.current_player == 1
        assert board.ko_point is None
        assert len(board.move_history) == 0
        assert board.captured_stones == {1: 0, 2: 0}
        
    def test_invalid_board_size(self):
        with pytest.raises(ValueError):
            Board(size=0)
        with pytest.raises(ValueError):
            Board(size=-1)
            
    def test_board_state(self, empty_board):
        assert np.all(empty_board.board == 0)
        assert isinstance(empty_board.board, np.ndarray)

class TestGameMechanics:
    def test_valid_moves(self, small_board):
        # Test valid move in empty position
        assert small_board.is_valid_move(4, 4)
        assert small_board.place_stone(4, 4)
        
        # Test invalid move on occupied position 
        assert not small_board.is_valid_move(4, 4)
        assert not small_board.place_stone(4, 4)
        
        # Test invalid moves outside board
        invalid_coords = [(-1, 0), (9, 0), (0, -1), (0, 9)]
        for x, y in invalid_coords:
            assert not small_board.is_valid_move(x, y)
            assert not small_board.place_stone(x, y)

    def test_alternating_turns(self, empty_board):
        assert empty_board.current_player == 1
        empty_board.place_stone(0, 0)
        assert empty_board.current_player == 2
        empty_board.place_stone(0, 1)
        assert empty_board.current_player == 1

    def test_move_history(self, empty_board):
        moves = [(0, 0), (0, 1), (1, 0)]
        for x, y in moves:
            empty_board.place_stone(x, y)
        assert empty_board.move_history == moves

class TestCaptureMechanics:
    def test_single_stone_capture(self, empty_board):
        # Black surrounds white stone
        empty_board.place_stone(1, 1)  # Black
        empty_board.place_stone(0, 1)  # White
        empty_board.place_stone(1, 0)  # Black
        empty_board.place_stone(1, 2)  # White being captured
        empty_board.place_stone(2, 1)  # Black completes capture
        
        assert empty_board.captured_stones[1] == 1
        assert empty_board.board[1, 2] == 0

    def test_group_capture(self, empty_board):
        # Setup white group
        white_stones = [(1, 1), (1, 2)]
        for x, y in white_stones:
            empty_board.current_player = 2
            empty_board.place_stone(x, y)
            
        # Black surrounds
        black_stones = [(0, 1), (0, 2), (1, 0), (1, 3), (2, 1), (2, 2)]
        for x, y in black_stones:
            empty_board.current_player = 1
            empty_board.place_stone(x, y)
            
        assert empty_board.captured_stones[1] == 2
        for x, y in white_stones:
            assert empty_board.board[x, y] == 0

    def test_self_capture_prevention(self, empty_board):
        # Setup position where move would be self-capture
        empty_board.place_stone(0, 1)  # Black
        empty_board.place_stone(1, 0)  # White
        empty_board.place_stone(1, 2)  # Black
        empty_board.place_stone(2, 1)  # White
        
        # Attempt self-capture at (1,1)
        assert not empty_board.is_valid_move(1, 1)

class TestKoRule:
    def test_basic_ko(self, empty_board):
        # Create ko situation
        moves = [
            (1, 1), (1, 2),  # Black, White
            (2, 2), (2, 1),  # Black, White
            (3, 1), (2, 0),  # Black, White
            (2, 1)           # Black captures
        ]
        for x, y in moves:
            empty_board.place_stone(x, y)
            
        # White shouldn't be able to recapture immediately
        assert not empty_board.is_valid_move(2, 1)
        
        # But can play elsewhere
        assert empty_board.is_valid_move(5, 5)
        
        # After another move, ko point should be cleared
        empty_board.place_stone(5, 5)
        assert empty_board.is_valid_move(2, 1)

class TestScoring:
    def test_territory_scoring(self, empty_board):
        # Create territories
        black_territory = [(0, 0), (0, 1), (1, 0)]
        white_territory = [(18, 18), (18, 17), (17, 18)]
        
        # Place black stones
        empty_board.current_player = 1
        for x, y in black_territory:
            empty_board.place_stone(x, y)
            
        # Place white stones
        empty_board.current_player = 2
        for x, y in white_territory:
            empty_board.place_stone(x, y)
            
        black_score, white_score = empty_board.get_score()
        assert black_score > len(black_territory)
        assert white_score > len(white_territory)
        
    def test_captured_stones_scoring(self, empty_board):
        # Simulate captures
        empty_board.captured_stones[1] = 5  # Black captured 5
        empty_board.captured_stones[2] = 3  # White captured 3
        
        black_score, white_score = empty_board.get_score()
        assert black_score >= 5
        assert white_score >= 3

    def test_empty_board_scoring(self, empty_board):
        black_score, white_score = empty_board.get_score()
        assert black_score == 0
        assert white_score == 0

class TestGameState:
    def test_board_copy(self, empty_board):
        empty_board.place_stone(0, 0)
        board_copy = empty_board.copy()
        
        assert np.array_equal(empty_board.board, board_copy.board)
        assert empty_board.current_player == board_copy.current_player
        assert empty_board.ko_point == board_copy.ko_point
        assert empty_board.captured_stones == board_copy.captured_stones
        
    def test_reset(self, empty_board):
        empty_board.place_stone(0, 0)
        empty_board.reset()
        
        assert np.all(empty_board.board == 0)
        assert empty_board.current_player == 1
        assert empty_board.ko_point is None
        assert len(empty_board.move_history) == 0
        assert empty_board.captured_stones == {1: 0, 2: 0}

    def test_get_valid_moves(self, empty_board):
        valid_moves = empty_board.get_valid_moves()
        assert valid_moves.shape == (empty_board.size, empty_board.size)
        assert np.all(valid_moves == 1)  # All moves valid on empty board
        
        empty_board.place_stone(0, 0)
        valid_moves = empty_board.get_valid_moves()
        assert valid_moves[0, 0] == 0  # Occupied position invalid