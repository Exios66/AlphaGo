import pytest
import torch
import numpy as np
from src.game.board import Board
from src.ai.policy_network import PolicyNetwork
from src.ai.value_network import ValueNetwork
from src.ai.mcts import MCTS, MCTSNode

@pytest.fixture
def board_size():
    return 19  # Standard Go board size

@pytest.fixture 
def policy_net(board_size):
    return PolicyNetwork(board_size=board_size)

@pytest.fixture
def value_net(board_size): 
    return ValueNetwork(board_size=board_size)

@pytest.fixture
def board(board_size):
    return Board(size=board_size)

@pytest.fixture
def mcts(policy_net, value_net):
    return MCTS(policy_net, value_net, num_simulations=800)

class TestPolicyNetwork:
    def test_initialization(self, board_size, policy_net):
        assert policy_net.board_size == board_size
        assert isinstance(policy_net, PolicyNetwork)
        
    def test_network_architecture(self, policy_net):
        # Test network layers and parameters
        assert hasattr(policy_net, 'conv_layers')
        assert hasattr(policy_net, 'fc_layers')
        assert next(policy_net.parameters()).requires_grad
        
    def test_input_preparation(self, board, policy_net):
        valid_moves = np.ones((board.size, board.size), dtype=np.float32)
        x = PolicyNetwork.prepare_input(board.board, board.current_player, valid_moves)
        
        assert isinstance(x, torch.Tensor)
        assert x.shape == (1, 3, board.size, board.size)
        assert not torch.isnan(x).any()
        
    def test_forward_pass(self, board, policy_net):
        valid_moves = np.ones((board.size, board.size), dtype=np.float32)
        x = PolicyNetwork.prepare_input(board.board, board.current_player, valid_moves)
        output = policy_net(x)
        
        assert output.shape == (1, board.size * board.size)
        assert torch.all(output <= 0)  # Log probabilities
        assert not torch.isnan(output).any()
        
    def test_move_prediction(self, board, policy_net):
        # Test empty board
        valid_moves = np.ones((board.size, board.size), dtype=np.float32)
        move = policy_net.predict_move(board.board, board.current_player, valid_moves)
        
        assert isinstance(move, tuple)
        assert len(move) == 2
        assert all(0 <= coord < board.size for coord in move)
        assert board.is_valid_move(*move)
        
        # Test with some stones on board
        board.place_stone(3, 3)
        board.place_stone(15, 15)
        move = policy_net.predict_move(board.board, board.current_player, valid_moves)
        
        assert isinstance(move, tuple)
        assert board.is_valid_move(*move)
        
    @pytest.mark.parametrize("invalid_input", [
        (None, 1, np.ones((19,19))),
        (np.zeros((19,19)), None, np.ones((19,19))),
        (np.zeros((19,19)), 1, None),
        (np.zeros((18,18)), 1, np.ones((19,19))),
    ])
    def test_invalid_inputs(self, policy_net, invalid_input):
        with pytest.raises((ValueError, TypeError)):
            policy_net.predict_move(*invalid_input)

class TestValueNetwork:
    def test_initialization(self, board_size, value_net):
        assert value_net.board_size == board_size
        assert isinstance(value_net, ValueNetwork)
        
    def test_network_architecture(self, value_net):
        assert hasattr(value_net, 'conv_layers')
        assert hasattr(value_net, 'fc_layers')
        assert next(value_net.parameters()).requires_grad
        
    def test_input_preparation(self, board, value_net):
        valid_moves = np.ones((board.size, board.size), dtype=np.float32)
        x = ValueNetwork.prepare_input(board.board, board.current_player, valid_moves)
        
        assert isinstance(x, torch.Tensor)
        assert x.shape == (1, 3, board.size, board.size)
        assert not torch.isnan(x).any()
        
    def test_forward_pass(self, board, value_net):
        valid_moves = np.ones((board.size, board.size), dtype=np.float32)
        x = ValueNetwork.prepare_input(board.board, board.current_player, valid_moves)
        output = value_net(x)
        
        assert output.shape == (1, 1)
        assert -1 <= output.item() <= 1
        assert not torch.isnan(output).any()
        
    def test_position_evaluation(self, board, value_net):
        # Test empty board
        valid_moves = np.ones((board.size, board.size), dtype=np.float32)
        value = value_net.evaluate_position(board.board, board.current_player, valid_moves)
        
        assert isinstance(value, float)
        assert -1 <= value <= 1
        
        # Test with some stones on board
        board.place_stone(3, 3)
        board.place_stone(15, 15)
        value = value_net.evaluate_position(board.board, board.current_player, valid_moves)
        
        assert isinstance(value, float)
        assert -1 <= value <= 1
        
    @pytest.mark.parametrize("invalid_input", [
        (None, 1, np.ones((19,19))),
        (np.zeros((19,19)), None, np.ones((19,19))),
        (np.zeros((19,19)), 1, None),
        (np.zeros((18,18)), 1, np.ones((19,19))),
    ])
    def test_invalid_inputs(self, value_net, invalid_input):
        with pytest.raises((ValueError, TypeError)):
            value_net.evaluate_position(*invalid_input)

class TestMCTS:
    def test_initialization(self, mcts, policy_net, value_net):
        assert isinstance(mcts.policy_net, PolicyNetwork)
        assert isinstance(mcts.value_net, ValueNetwork)
        assert mcts.num_simulations == 800
        
    def test_mcts_node(self, board):
        node = MCTSNode(board)
        assert node.visit_count == 0
        assert node.value() == 0.0
        assert not node.is_expanded()
        assert node.children == {}
        
    def test_selection(self, mcts, board):
        node = MCTSNode(board)
        mcts._expand(node)
        selected_node = mcts._select(node)
        
        assert isinstance(selected_node, MCTSNode)
        assert selected_node.visit_count == 0
        
    def test_expansion(self, mcts, board):
        node = MCTSNode(board)
        mcts._expand(node)
        
        assert node.is_expanded()
        assert len(node.children) > 0
        for child in node.children.values():
            assert isinstance(child, MCTSNode)
            
    def test_simulation(self, mcts, board):
        node = MCTSNode(board)
        value = mcts._simulate(node)
        
        assert isinstance(value, float)
        assert -1 <= value <= 1
        
    def test_backpropagation(self, mcts, board):
        node = MCTSNode(board)
        mcts._expand(node)
        child = list(node.children.values())[0]
        mcts._backpropagate(child, 0.5)
        
        assert node.visit_count == 1
        assert child.visit_count == 1
        
    def test_move_selection(self, mcts, board):
        # Test on empty board
        move = mcts.select_move(board)
        assert isinstance(move, tuple)
        assert len(move) == 2
        assert all(0 <= coord < board.size for coord in move)
        assert board.is_valid_move(*move)
        
        # Test with some stones on board
        board.place_stone(3, 3)
        board.place_stone(15, 15)
        move = mcts.select_move(board)
        
        assert isinstance(move, tuple)
        assert board.is_valid_move(*move)
        
    def test_parallel_game_tree_search(self, mcts, board):
        # Test running multiple simulations in parallel
        moves = [mcts.select_move(board) for _ in range(3)]
        
        assert len(moves) == 3
        assert all(isinstance(m, tuple) for m in moves)
        assert all(board.is_valid_move(*m) for m in moves)
        
    @pytest.mark.parametrize("num_moves", [1, 5, 10])
    def test_game_playthrough(self, mcts, board, num_moves):
        # Test playing multiple moves
        for _ in range(num_moves):
            move = mcts.select_move(board)
            assert board.is_valid_move(*move)
            board.place_stone(*move)
            
        assert len(board.move_history) == num_moves