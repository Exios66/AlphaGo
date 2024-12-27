import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class PolicyNetwork(nn.Module):
    """
    Policy Network that predicts move probabilities for the Go board.
    Architecture inspired by AlphaGo's policy network, but simplified.
    """
    
    def __init__(self, board_size: int = 19, num_channels: int = 256):
        """
        Initialize the policy network.
        
        Args:
            board_size (int): Size of the Go board
            num_channels (int): Number of channels in convolutional layers
        """
        super(PolicyNetwork, self).__init__()
        
        # Input channels:
        # 1: Current player stones (1s)
        # 2: Opponent stones (1s)
        # 3: Valid moves (1s)
        # 4: Ones if current player is black, zeros if white
        self.num_input_channels = 4
        
        # Network structure
        self.conv1 = nn.Conv2d(self.num_input_channels, num_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.conv4 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.conv5 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.conv6 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.conv7 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.conv8 = nn.Conv2d(num_channels, 1, 1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.bn3 = nn.BatchNorm2d(num_channels)
        self.bn4 = nn.BatchNorm2d(num_channels)
        self.bn5 = nn.BatchNorm2d(num_channels)
        self.bn6 = nn.BatchNorm2d(num_channels)
        self.bn7 = nn.BatchNorm2d(num_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_input_channels, board_size, board_size)
            
        Returns:
            torch.Tensor: Move probabilities of shape (batch_size, board_size * board_size)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.conv8(x)
        
        # Flatten the output and apply softmax
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        return F.log_softmax(x, dim=1)
    
    @staticmethod
    def prepare_input(board: np.ndarray, current_player: int, valid_moves: np.ndarray) -> torch.Tensor:
        """
        Prepare the input tensor for the network from a board state.
        
        Args:
            board (np.ndarray): The game board
            current_player (int): Current player (1 for black, 2 for white)
            valid_moves (np.ndarray): Binary array indicating valid moves
            
        Returns:
            torch.Tensor: Input tensor for the network
        """
        board_size = board.shape[0]
        input_tensor = np.zeros((4, board_size, board_size), dtype=np.float32)
        
        # Channel 1: Current player's stones
        input_tensor[0] = (board == current_player)
        
        # Channel 2: Opponent's stones
        input_tensor[1] = (board == 3 - current_player)
        
        # Channel 3: Valid moves
        input_tensor[2] = valid_moves
        
        # Channel 4: Current player color (1 for black, 0 for white)
        input_tensor[3] = np.full((board_size, board_size), current_player == 1)
        
        return torch.FloatTensor(input_tensor).unsqueeze(0)
    
    def predict_move(self, board: np.ndarray, current_player: int, valid_moves: np.ndarray) -> Tuple[int, int]:
        """
        Predict the best move given the current board state.
        
        Args:
            board (np.ndarray): The game board
            current_player (int): Current player (1 for black, 2 for white)
            valid_moves (np.ndarray): Binary array indicating valid moves
            
        Returns:
            Tuple[int, int]: The predicted move coordinates (x, y)
        """
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            x = self.prepare_input(board, current_player, valid_moves)
            output = self.forward(x)
            move_probs = torch.exp(output).squeeze()
            
            # Mask invalid moves
            valid_moves_flat = torch.FloatTensor(valid_moves.flatten())
            move_probs *= valid_moves_flat
            
            # Select the move with highest probability
            move_idx = torch.argmax(move_probs).item()
            board_size = board.shape[0]
            return move_idx // board_size, move_idx % board_size 