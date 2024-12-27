import numpy as np
from typing import Dict, List, Optional, Tuple
import math
from copy import deepcopy
import torch

from src.game.board import Board
from src.ai.policy_network import PolicyNetwork
from src.ai.value_network import ValueNetwork

class MCTSNode:
    """
    Node in the Monte Carlo Tree Search.
    """
    
    def __init__(self, board: Board, parent: Optional['MCTSNode'] = None, move: Optional[Tuple[int, int]] = None):
        self.board = deepcopy(board)
        self.parent = parent
        self.move = move
        self.children: Dict[Tuple[int, int], MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_probability = 0.0
        
    def value(self) -> float:
        """Get the average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
        
    def is_expanded(self) -> bool:
        """Check if the node has been expanded."""
        return len(self.children) > 0

class MCTS:
    """
    Monte Carlo Tree Search implementation that uses policy and value networks
    for tree search guidance and position evaluation.
    """
    
    def __init__(self, policy_network: PolicyNetwork, value_network: ValueNetwork,
                 num_simulations: int = 800, c_puct: float = 1.0):
        """
        Initialize MCTS.
        
        Args:
            policy_network: Network for move prediction
            value_network: Network for position evaluation
            num_simulations: Number of simulations to run
            c_puct: Exploration constant
        """
        self.policy_network = policy_network
        self.value_network = value_network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        
    def select_move(self, board: Board) -> Tuple[int, int]:
        """
        Select the best move using MCTS.
        
        Args:
            board: Current game board
            
        Returns:
            Tuple[int, int]: Best move coordinates
        """
        root = MCTSNode(board)
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Selection: traverse tree to leaf node
            while node.is_expanded():
                node = self._select_child(node)
                search_path.append(node)
                
            # Expansion and evaluation
            value = self._expand_and_evaluate(node)
            
            # Backpropagation
            self._backpropagate(search_path, value)
            
        # Select move with highest visit count
        return self._select_best_move(root)
        
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """
        Select the child node with the highest UCT value.
        
        Args:
            node: Current node
            
        Returns:
            MCTSNode: Selected child node
        """
        best_score = float('-inf')
        best_child = None
        total_visits = sum(child.visit_count for child in node.children.values())
        
        for move, child in node.children.items():
            # UCT formula with prior probability
            exploration = self.c_puct * child.prior_probability * math.sqrt(total_visits) / (1 + child.visit_count)
            uct_score = child.value() + exploration
            
            if uct_score > best_score:
                best_score = uct_score
                best_child = child
                
        return best_child
        
    def _expand_and_evaluate(self, node: MCTSNode) -> float:
        """
        Expand the node and evaluate its position.
        
        Args:
            node: Node to expand
            
        Returns:
            float: Position evaluation
        """
        # Get valid moves
        valid_moves = np.zeros((node.board.size, node.board.size), dtype=np.float32)
        legal_moves = node.board.get_legal_moves()
        for x, y in legal_moves:
            valid_moves[x, y] = 1
            
        # Get move probabilities from policy network
        board_tensor = PolicyNetwork.prepare_input(
            node.board.board,
            node.board.current_player,
            valid_moves
        )
        with torch.no_grad():
            move_probs = torch.exp(self.policy_network(board_tensor)).squeeze()
            
        # Create children for each legal move
        for x, y in legal_moves:
            if (x, y) not in node.children:
                new_board = deepcopy(node.board)
                new_board.place_stone(x, y)
                child = MCTSNode(new_board, parent=node, move=(x, y))
                child.prior_probability = move_probs[x * node.board.size + y].item()
                node.children[(x, y)] = child
                
        # Evaluate position
        value = self.value_network.evaluate_position(
            node.board.board,
            node.board.current_player,
            valid_moves
        )
        
        return value
        
    def _backpropagate(self, search_path: List[MCTSNode], value: float):
        """
        Backpropagate the evaluation through the tree.
        
        Args:
            search_path: Path of nodes traversed
            value: Position evaluation
        """
        for node in search_path:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip value for opponent
            
    def _select_best_move(self, root: MCTSNode) -> Tuple[int, int]:
        """
        Select the best move based on visit counts.
        
        Args:
            root: Root node
            
        Returns:
            Tuple[int, int]: Best move coordinates
        """
        visits = {move: child.visit_count for move, child in root.children.items()}
        return max(visits.items(), key=lambda x: x[1])[0] 