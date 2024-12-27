import os
import gzip
import urllib.request
import numpy as np
from typing import List, Tuple, Generator
from sgfmill import sgf
from tqdm import tqdm

class GoDataProcessor:
    """Process Go game records for training."""
    
    def __init__(self, data_dir: str = "data/datasets"):
        """
        Initialize the data processor.
        
        Args:
            data_dir: Directory to store downloaded and processed data
        """
        self.data_dir = data_dir
        self.kgs_url = "https://u-go.net/gamerecords/"
        self.processed_dir = os.path.join(data_dir, "processed")
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        
    def download_kgs_games(self, num_games: int = 1000):
        """
        Download Go games from KGS archives.
        
        Args:
            num_games: Number of games to download
        """
        print(f"Downloading {num_games} games from KGS...")
        
        # Download index page to find game archive links
        index = urllib.request.urlopen(self.kgs_url).read().decode('utf-8')
        archive_links = [line.split('"')[1] for line in index.split('\n') if '.tar.gz' in line]
        
        games_downloaded = 0
        for link in archive_links:
            if games_downloaded >= num_games:
                break
                
            archive_path = os.path.join(self.data_dir, os.path.basename(link))
            if not os.path.exists(archive_path):
                print(f"Downloading {link}...")
                urllib.request.urlretrieve(self.kgs_url + link, archive_path)
            
            # Extract games from archive
            with gzip.open(archive_path, 'rb') as f:
                games = f.read().decode('utf-8').split('\n\n')
                for game in games:
                    if games_downloaded >= num_games:
                        break
                    if game.strip():
                        game_path = os.path.join(self.data_dir, f"game_{games_downloaded}.sgf")
                        with open(game_path, 'w') as gf:
                            gf.write(game)
                        games_downloaded += 1
                        
        print(f"Downloaded {games_downloaded} games")
        
    def process_game(self, sgf_content: str) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Process a single game record.
        
        Args:
            sgf_content: SGF game record content
            
        Returns:
            List of (board_state, next_move, outcome) tuples
        """
        game = sgf.Sgf_game.from_string(sgf_content.encode('utf-8'))
        board_size = 19  # Standard board size
        
        # Initialize empty board
        board = np.zeros((board_size, board_size), dtype=np.int8)
        current_player = 1  # Black starts
        training_data = []
        
        # Get game outcome
        try:
            result = game.root.properties['RE'][0]
            if 'B+' in result:
                outcome = 1.0  # Black win
            elif 'W+' in result:
                outcome = -1.0  # White win
            else:
                return []  # Skip games with unknown outcome
        except:
            return []  # Skip games with no result
            
        # Process moves
        for node in game.rest:
            try:
                # Get move coordinates
                if 'B' in node.properties:
                    move = node.properties['B'][0]
                    player = 1
                elif 'W' in node.properties:
                    move = node.properties['W'][0]
                    player = 2
                else:
                    continue
                    
                if move == '' or move == 'tt':  # Pass move
                    continue
                    
                x = ord(move[0]) - ord('a')
                y = ord(move[1]) - ord('a')
                
                if not (0 <= x < board_size and 0 <= y < board_size):
                    continue
                    
                # Create training example
                board_state = board.copy()
                move_matrix = np.zeros((board_size, board_size), dtype=np.float32)
                move_matrix[x, y] = 1
                
                # Flip outcome for white's moves
                example_outcome = outcome if player == 1 else -outcome
                
                training_data.append((board_state, move_matrix, example_outcome))
                
                # Update board state
                board[x, y] = player
                
            except Exception as e:
                print(f"Error processing move: {e}")
                continue
                
        return training_data
        
    def prepare_training_data(self, num_games: int = None) -> Generator[Tuple[np.ndarray, np.ndarray, float], None, None]:
        """
        Prepare training data from processed games.
        
        Args:
            num_games: Number of games to process (None for all games)
            
        Returns:
            Generator yielding (board_state, next_move, outcome) tuples
        """
        game_files = [f for f in os.listdir(self.data_dir) if f.endswith('.sgf')]
        if num_games is not None:
            game_files = game_files[:num_games]
            
        print(f"Processing {len(game_files)} games...")
        for game_file in tqdm(game_files):
            try:
                with open(os.path.join(self.data_dir, game_file), 'r') as f:
                    sgf_content = f.read()
                    
                training_examples = self.process_game(sgf_content)
                for example in training_examples:
                    yield example
                    
            except Exception as e:
                print(f"Error processing {game_file}: {e}")
                continue 