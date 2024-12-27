import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple
import os
from tqdm import tqdm

from src.ai.policy_network import PolicyNetwork
from src.utils.data_processing import GoDataProcessor

class GoDataset(Dataset):
    """Dataset for training the policy network."""
    
    def __init__(self, examples: List[Tuple[np.ndarray, np.ndarray, float]]):
        """
        Initialize the dataset.
        
        Args:
            examples: List of (board_state, next_move, outcome) tuples
        """
        self.examples = examples
        
    def __len__(self) -> int:
        return len(self.examples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        board, move, _ = self.examples[idx]
        
        # Prepare input features (4 channels)
        features = np.zeros((4, board.shape[0], board.shape[1]), dtype=np.float32)
        features[0] = (board == 1)  # Black stones
        features[1] = (board == 2)  # White stones
        features[2] = np.ones_like(board)  # Valid moves (simplified)
        features[3] = np.ones_like(board)  # Current player is black
        
        # Flatten move matrix for cross-entropy loss
        move_target = move.flatten()
        
        return torch.FloatTensor(features), torch.FloatTensor(move_target)

def train_policy_network(
    board_size: int = 19,
    num_games: int = 1000,
    batch_size: int = 32,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train the policy network using supervised learning.
    
    Args:
        board_size: Size of the Go board
        num_games: Number of games to use for training
        batch_size: Training batch size
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        device: Device to use for training
    """
    print(f"Training policy network on {device}...")
    
    # Initialize network
    policy_net = PolicyNetwork(board_size=board_size)
    policy_net.to(device)
    
    # Load existing model if available
    model_path = "data/models/policy_net.pth"
    if os.path.exists(model_path):
        print("Loading existing model...")
        policy_net.load_state_dict(torch.load(model_path))
    
    # Prepare training data
    print("Preparing training data...")
    data_processor = GoDataProcessor()
    data_processor.download_kgs_games(num_games)
    training_examples = list(data_processor.prepare_training_data(num_games))
    
    if not training_examples:
        print("No training examples found!")
        return
    
    # Create data loader
    dataset = GoDataset(training_examples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(num_epochs):
        policy_net.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, targets in progress_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = policy_net(inputs)
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
            torch.save(policy_net.state_dict(), model_path)
            
    print("Training completed!")

if __name__ == "__main__":
    train_policy_network() 