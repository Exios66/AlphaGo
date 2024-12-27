import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple
import os
from tqdm import tqdm
import random

from src.game.board import Board
from src.ai.value_network import ValueNetwork
from src.ai.policy_network import PolicyNetwork
from src.ai.mcts import MCTS

class SelfPlayDataset(Dataset):
    """Dataset for training the value network using self-play data."""
    
    def __init__(self, examples: List[Tuple[np.ndarray, int, float]]):
        """
        Initialize the dataset.
        
        Args:
            examples: List of (board_state, current_player, outcome) tuples
        """
        self.examples = examples
        
    def __len__(self) -> int:
        return len(self.examples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        board, player, outcome = self.examples[idx]
        
        # Prepare input features (4 channels)
        features = np.zeros((4, board.shape[0], board.shape[1]), dtype=np.float32)
        features[0] = (board == player)  # Current player stones
        features[1] = (board == 3 - player)  # Opponent stones
        features[2] = np.ones_like(board)  # Valid moves (simplified)
        features[3] = np.full_like(board, player == 1)  # Current player is black
        
        return torch.FloatTensor(features), torch.FloatTensor([outcome])

def generate_self_play_data(
    num_games: int,
    board_size: int,
    policy_net: PolicyNetwork,
    mcts_simulations: int = 100
) -> List[Tuple[np.ndarray, int, float]]:
    """
    Generate training data through self-play.
    
    Args:
        num_games: Number of self-play games to generate
        board_size: Size of the Go board
        policy_net: Trained policy network for MCTS
        mcts_simulations: Number of MCTS simulations per move
        
    Returns:
        List of (board_state, current_player, outcome) tuples
    """
    print(f"Generating {num_games} self-play games...")
    examples = []
    
    # Initialize temporary value network for MCTS
    temp_value_net = ValueNetwork(board_size)
    mcts = MCTS(policy_net, temp_value_net, num_simulations=mcts_simulations)
    
    for game_idx in tqdm(range(num_games)):
        board = Board(board_size)
        game_states = []
        
        # Play until game is over or move limit reached
        move_count = 0
        max_moves = board_size * board_size * 2  # Reasonable maximum
        
        while move_count < max_moves:
            current_state = board.board.copy()
            current_player = board.current_player
            
            # Get move from MCTS
            move = mcts.select_move(board)
            if not board.place_stone(*move):
                break
                
            # Store state
            game_states.append((current_state, current_player))
            move_count += 1
            
            # Check if game is over
            if len(board.get_legal_moves()) == 0:
                break
                
        # Calculate game outcome
        black_score, white_score = board.get_score()
        outcome = 1.0 if black_score > white_score else -1.0
        
        # Add all game states to training examples
        for state, player in game_states:
            # Flip outcome for white's moves
            state_outcome = outcome if player == 1 else -outcome
            examples.append((state, player, state_outcome))
            
    return examples

def train_value_network(
    board_size: int = 19,
    num_games: int = 100,
    batch_size: int = 32,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train the value network using self-play data.
    
    Args:
        board_size: Size of the Go board
        num_games: Number of self-play games to generate
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        device: Device to use for training
    """
    print(f"Training value network on {device}...")
    
    # Initialize networks
    value_net = ValueNetwork(board_size=board_size)
    value_net.to(device)
    
    # Load policy network for self-play
    policy_net = PolicyNetwork(board_size=board_size)
    policy_path = "data/models/policy_net.pth"
    if os.path.exists(policy_path):
        policy_net.load_state_dict(torch.load(policy_path))
    policy_net.to(device)
    policy_net.eval()
    
    # Load existing value network if available
    model_path = "data/models/value_net.pth"
    if os.path.exists(model_path):
        print("Loading existing model...")
        value_net.load_state_dict(torch.load(model_path))
    
    # Generate training data through self-play
    training_examples = generate_self_play_data(num_games, board_size, policy_net)
    
    if not training_examples:
        print("No training examples generated!")
        return
    
    # Create data loader
    dataset = SelfPlayDataset(training_examples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = optim.Adam(value_net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(num_epochs):
        value_net.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, targets in progress_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = value_net(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({'loss': total_loss / num_batches})
            
        # Calculate average loss
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"Saving improved model (loss: {best_loss:.4f})...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(value_net.state_dict(), model_path)
            
    print("Training completed!")

if __name__ == "__main__":
    train_value_network() 