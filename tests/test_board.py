import pytest
from src.game.board import Board

def test_board_initialization():
    board = Board(size=19)
    assert board.size == 19
    assert board.current_player == 1
    assert board.ko_point is None
    assert len(board.move_history) == 0
    assert board.captured_stones == {1: 0, 2: 0}

def test_valid_moves():
    board = Board(size=9)
    # Test valid move in empty position
    assert board.is_valid_move(4, 4)
    assert board.place_stone(4, 4)
    # Test invalid move on occupied position
    assert not board.is_valid_move(4, 4)
    assert not board.place_stone(4, 4)
    # Test invalid move outside board
    assert not board.is_valid_move(-1, 0)
    assert not board.is_valid_move(9, 0)

def test_capture():
    board = Board(size=5)
    # Create a capture scenario
    # Black stones
    board.place_stone(1, 0)
    board.place_stone(1, 2)
    # White stones
    board.place_stone(1, 1)
    board.place_stone(0, 1)
    # Black captures white stones
    board.place_stone(2, 1)
    assert board.captured_stones[1] == 2  # Black captured 2 white stones
    assert board.board[1, 1] == 0  # Captured stone removed
    assert board.board[0, 1] == 0  # Captured stone removed

def test_ko_rule():
    board = Board(size=5)
    # Create a ko scenario
    # Black stones
    board.place_stone(1, 1)
    board.place_stone(2, 2)
    # White stones
    board.place_stone(2, 1)
    board.place_stone(1, 3)
    board.place_stone(2, 3)
    # Black captures
    board.place_stone(2, 2)
    # White shouldn't be able to recapture immediately (ko rule)
    assert not board.is_valid_move(2, 2)

def test_scoring():
    board = Board(size=5)
    # Create a simple territory scenario
    # Black stones
    board.place_stone(0, 0)
    board.place_stone(0, 1)
    board.place_stone(1, 0)
    # White stones
    board.place_stone(3, 3)
    board.place_stone(3, 4)
    board.place_stone(4, 3)
    board.place_stone(4, 4)
    
    black_score, white_score = board.get_score()
    assert black_score > 3  # Should include territory
    assert white_score > 4  # Should include territory 