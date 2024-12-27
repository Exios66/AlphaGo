import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ValueNetwork(nn.Module):
    """
    Value Network that evaluates board positions.
    Returns a value between -1 and 1, where:
    - 1 means current player is very likely to win
    - -1 means current player is very likely to lose
    """
    
    def __init__(self, board_size: int = 19, num_channels: int = 256):
        """
        Initialize the value network.
        
        Args:
            board_size (int): Size of the Go board
            num_channels (int): Number of channels in convolutional layers
        """
        super(ValueNetwork, self).__init__()
        
        # Input channels (same as policy network):
        # 1: Current player stones (1s)
        # 2: Opponent stones (1s)
        # 3: Valid moves (1s)
        # 4: Ones if current player is black, zeros if white
        self.num_input_channels = 4
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(self.num_input_channels, num_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.conv4 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.conv5 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.bn3 = nn.BatchNorm2d(num_channels)
        self.bn4 = nn.BatchNorm2d(num_channels)
        self.bn5 = nn.BatchNorm2d(num_channels)
        
        # Fully connected layers
        self.fc_input_size = num_channels * board_size * board_size
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_input_channels, board_size, board_size)
            
        Returns:
            torch.Tensor: Value prediction between -1 and 1
        """
        # Convolutional layers with batch normalization and ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Flatten and fully connected layers
        x = x.view(-1, self.fc_input_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # Tanh activation to get value between -1 and 1
        return torch.tanh(x)
    
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
    
    def evaluate_position(self, board: np.ndarray, current_player: int, valid_moves: np.ndarray) -> float:
        """
        Evaluate the current board position.
        
        Args:
            board (np.ndarray): The game board
            current_player (int): Current player (1 for black, 2 for white)
            valid_moves (np.ndarray): Binary array indicating valid moves
            
        Returns:
            float: Position evaluation between -1 and 1
        """
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            x = self.prepare_input(board, current_player, valid_moves)
            value = self.forward(x)
            return value.item() 