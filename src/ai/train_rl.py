import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple
import os
from tqdm import tqdm
import random

from src.game.board import Board
from src.ai.policy_network import PolicyNetwork
from src.ai.value_network import ValueNetwork
from src.ai.mcts import MCTS

class RLTrainer:
    """Reinforcement Learning trainer for both networks."""
    
    def __init__(
        self,
        board_size: int = 19,
        mcts_simulations: int = 800,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the RL trainer.
        
        Args:
            board_size: Size of the Go board
            mcts_simulations: Number of MCTS simulations per move
            device: Device to use for training
        """
        self.board_size = board_size
        self.device = device
        self.mcts_simulations = mcts_simulations
        
        # Initialize networks
        self.policy_net = PolicyNetwork(board_size).to(device)
        self.value_net = ValueNetwork(board_size).to(device)
        
        # Load pre-trained models if available
        if os.path.exists("data/models/policy_net.pth"):
            self.policy_net.load_state_dict(torch.load("data/models/policy_net.pth"))
        if os.path.exists("data/models/value_net.pth"):
            self.value_net.load_state_dict(torch.load("data/models/value_net.pth"))
            
        # Initialize MCTS
        self.mcts = MCTS(self.policy_net, self.value_net, num_simulations=mcts_simulations)
        
    def play_game(self) -> List[Tuple[np.ndarray, np.ndarray, int, float]]:
        """
        Play a single game and return the training examples.
        
        Returns:
            List of (board_state, policy_target, current_player, outcome) tuples
        """
        board = Board(self.board_size)
        examples = []
        
        while True:
            current_state = board.board.copy()
            current_player = board.current_player
            
            # Get valid moves
            valid_moves = np.zeros((self.board_size, self.board_size), dtype=np.float32)
            legal_moves = board.get_legal_moves()
            for x, y in legal_moves:
                valid_moves[x, y] = 1
                
            # Use MCTS to get improved policy
            root = self.mcts.select_move(board, return_probabilities=True)
            policy = np.zeros((self.board_size, self.board_size), dtype=np.float32)
            
            # Convert visit counts to policy
            total_visits = sum(child.visit_count for child in root.children.values())
            for move, child in root.children.items():
                x, y = move
                policy[x, y] = child.visit_count / total_visits
                
            # Store state and policy
            examples.append((current_state, policy, current_player, None))
            
            # Make move
            move = self.mcts.select_move(board)
            if not board.place_stone(*move):
                break
                
            # Check if game is over
            if len(board.get_legal_moves()) == 0:
                break
                
        # Calculate game outcome
        black_score, white_score = board.get_score()
        outcome = 1.0 if black_score > white_score else -1.0
        
        # Update examples with actual outcome
        examples = [(state, policy, player, outcome if player == 1 else -outcome)
                   for state, policy, player, _ in examples]
                   
        return examples
        
    def train_iteration(
        self,
        num_games: int = 25,
        batch_size: int = 32,
        num_epochs: int = 5,
        policy_lr: float = 0.0001,
        value_lr: float = 0.0001
    ):
        """
        Perform one iteration of reinforcement learning.
        
        Args:
            num_games: Number of self-play games per iteration
            batch_size: Training batch size
            num_epochs: Number of training epochs
            policy_lr: Learning rate for policy network
            value_lr: Learning rate for value network
        """
        print(f"Starting training iteration with {num_games} games...")
        
        # Generate training data through self-play
        examples = []
        for _ in tqdm(range(num_games), desc="Self-play games"):
            game_examples = self.play_game()
            examples.extend(game_examples)
            
        if not examples:
            print("No training examples generated!")
            return
            
        # Convert examples to tensors
        states = []
        policies = []
        outcomes = []
        
        for state, policy, player, outcome in examples:
            # Prepare input features
            features = np.zeros((4, self.board_size, self.board_size), dtype=np.float32)
            features[0] = (state == player)
            features[1] = (state == 3 - player)
            features[2] = np.ones_like(state)
            features[3] = np.full_like(state, player == 1)
            
            states.append(features)
            policies.append(policy.flatten())
            outcomes.append(outcome)
            
        states = torch.FloatTensor(np.array(states)).to(self.device)
        policies = torch.FloatTensor(np.array(policies)).to(self.device)
        outcomes = torch.FloatTensor(np.array(outcomes)).to(self.device)
        
        # Training
        dataset_size = len(examples)
        indices = list(range(dataset_size))
        
        policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        value_optimizer = optim.Adam(self.value_net.parameters(), lr=value_lr)
        policy_criterion = nn.CrossEntropyLoss()
        value_criterion = nn.MSELoss()
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            random.shuffle(indices)
            
            total_policy_loss = 0.0
            total_value_loss = 0.0
            num_batches = 0
            
            for i in range(0, dataset_size, batch_size):
                batch_indices = indices[i:min(i + batch_size, dataset_size)]
                batch_states = states[batch_indices]
                batch_policies = policies[batch_indices]
                batch_outcomes = outcomes[batch_indices].unsqueeze(1)
                
                # Train policy network
                self.policy_net.train()
                policy_optimizer.zero_grad()
                policy_out = self.policy_net(batch_states)
                policy_loss = policy_criterion(policy_out, batch_policies)
                policy_loss.backward()
                policy_optimizer.step()
                
                # Train value network
                self.value_net.train()
                value_optimizer.zero_grad()
                value_out = self.value_net(batch_states)
                value_loss = value_criterion(value_out, batch_outcomes)
                value_loss.backward()
                value_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1
                
            avg_policy_loss = total_policy_loss / num_batches
            avg_value_loss = total_value_loss / num_batches
            print(f"Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}")
            
        # Save improved models
        print("Saving improved models...")
        os.makedirs("data/models", exist_ok=True)
        torch.save(self.policy_net.state_dict(), "data/models/policy_net.pth")
        torch.save(self.value_net.state_dict(), "data/models/value_net.pth")

def train_rl(
    num_iterations: int = 10,
    games_per_iteration: int = 25,
    board_size: int = 19
):
    """
    Perform reinforcement learning training.
    
    Args:
        num_iterations: Number of training iterations
        games_per_iteration: Number of games per iteration
        board_size: Size of the Go board
    """
    trainer = RLTrainer(board_size=board_size)
    
    for iteration in range(num_iterations):
        print(f"\nStarting iteration {iteration+1}/{num_iterations}")
        trainer.train_iteration(num_games=games_per_iteration)
        print(f"Completed iteration {iteration+1}")

if __name__ == "__main__":
    train_rl() 