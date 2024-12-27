import pytest
import torch
import numpy as np
from src.game.board import Board
from src.ai.policy_network import PolicyNetwork
from src.ai.value_network import ValueNetwork
from src.ai.mcts import MCTS, MCTSNode

def test_policy_network():
    # Initialize network
    board_size = 9
    policy_net = PolicyNetwork(board_size=board_size)
    
    # Create a simple board state
    board = Board(size=board_size)
    board.place_stone(2, 2)  # Black stone
    board.place_stone(3, 3)  # White stone
    
    # Get valid moves
    valid_moves = np.zeros((board_size, board_size), dtype=np.float32)
    legal_moves = board.get_legal_moves()
    for x, y in legal_moves:
        valid_moves[x, y] = 1
    
    # Test forward pass
    x = PolicyNetwork.prepare_input(board.board, board.current_player, valid_moves)
    output = policy_net(x)
    
    # Check output shape and properties
    assert output.shape == (1, board_size * board_size)
    assert torch.all(output <= 0)  # Log probabilities should be <= 0
    
    # Test move prediction
    move = policy_net.predict_move(board.board, board.current_player, valid_moves)
    assert isinstance(move, tuple)
    assert len(move) == 2
    assert 0 <= move[0] < board_size
    assert 0 <= move[1] < board_size

def test_value_network():
    # Initialize network
    board_size = 9
    value_net = ValueNetwork(board_size=board_size)
    
    # Create a simple board state
    board = Board(size=board_size)
    board.place_stone(2, 2)  # Black stone
    board.place_stone(3, 3)  # White stone
    
    # Get valid moves
    valid_moves = np.zeros((board_size, board_size), dtype=np.float32)
    legal_moves = board.get_legal_moves()
    for x, y in legal_moves:
        valid_moves[x, y] = 1
    
    # Test forward pass
    x = ValueNetwork.prepare_input(board.board, board.current_player, valid_moves)
    output = value_net(x)
    
    # Check output shape and properties
    assert output.shape == (1, 1)
    assert -1 <= output.item() <= 1  # Value should be between -1 and 1
    
    # Test position evaluation
    value = value_net.evaluate_position(board.board, board.current_player, valid_moves)
    assert isinstance(value, float)
    assert -1 <= value <= 1

def test_mcts():
    # Initialize components
    board_size = 9
    policy_net = PolicyNetwork(board_size=board_size)
    value_net = ValueNetwork(board_size=board_size)
    mcts = MCTS(policy_net, value_net, num_simulations=10)  # Small number for testing
    
    # Create a simple board state
    board = Board(size=board_size)
    board.place_stone(2, 2)  # Black stone
    board.place_stone(3, 3)  # White stone
    
    # Test MCTS node
    node = MCTSNode(board)
    assert node.visit_count == 0
    assert node.value() == 0.0
    assert not node.is_expanded()
    
    # Test move selection
    move = mcts.select_move(board)
    assert isinstance(move, tuple)
    assert len(move) == 2
    assert 0 <= move[0] < board_size
    assert 0 <= move[1] < board_size
    assert board.is_valid_move(*move) 