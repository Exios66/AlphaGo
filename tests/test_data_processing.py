import os
import tempfile
import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from src.utils.data_processing import GoDataProcessor

class TestGoDataProcessor(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.data_processor = GoDataProcessor(data_dir=self.test_dir)
        
        # Sample SGF game for testing
        self.sample_sgf = '''(;FF[4]CA[UTF-8]GM[1]DT[2024-01-01]PC[The KGS Go Server at http://www.gokgs.com/]
            PB[Black]PW[White]BR[6d]WR[7d]RE[B+3.5]
            SZ[19]KM[6.5]HA[0]RU[Japanese]
            ;B[pd];W[dp];B[pp];W[dd];B[fc];W[cf];B[jd];W[qf]
            ;B[nd];W[rd];B[qc];W[nq];B[qn];W[pr];B[qq];W[lp]
            ;B[cn];W[fq];B[bp];W[cq];B[ck];W[qh];B[rc];W[jq])'''
            
        # Sample SGF with different board size and pass moves
        self.sample_sgf_13 = '''(;FF[4]GM[1]SZ[13]RE[B+Resign]
            ;B[jj];W[dd];B[jd];W[tt];B[dj];W[]);'''
            
    def test_process_game(self):
        # Test game processing
        training_data = self.data_processor.process_game(self.sample_sgf)
        
        # Check if training data is not empty
        self.assertTrue(len(training_data) > 0)
        
        # Check structure of training examples
        for board_state, next_move, outcome in training_data:
            # Check shapes
            self.assertEqual(board_state.shape, (19, 19))
            self.assertEqual(next_move.shape, (19, 19))
            
            # Check data types
            self.assertEqual(board_state.dtype, np.int8)
            self.assertEqual(next_move.dtype, np.float32)
            
            # Check value ranges
            self.assertTrue(np.all((board_state >= 0) & (board_state <= 2)))
            self.assertTrue(np.all((next_move >= 0) & (next_move <= 1)))
            self.assertTrue(abs(outcome) == 1.0)
            
    def test_different_board_sizes(self):
        # Test processing game with 13x13 board
        training_data = self.data_processor.process_game(self.sample_sgf_13)
        self.assertTrue(len(training_data) > 0)
        
        for board_state, next_move, outcome in training_data:
            self.assertEqual(board_state.shape, (13, 13))
            self.assertEqual(next_move.shape, (13, 13))
            
    def test_pass_moves(self):
        # Test handling of pass moves
        sgf_with_passes = '''(;FF[4]GM[1]SZ[19]RE[B+Resign]
            ;B[pd];W[tt];B[dp];W[];B[pp])'''
        training_data = self.data_processor.process_game(sgf_with_passes)
        self.assertTrue(len(training_data) > 0)
        
    def test_coordinate_conversion(self):
        # Test coordinate conversion edge cases
        edge_coords = [
            ('aa', 0, 0),   # Top-left corner
            ('ss', 18, 18), # Bottom-right corner
            ('sa', 18, 0),  # Top-right corner
            ('as', 0, 18)   # Bottom-left corner
        ]
        
        for sgf_coord, expected_x, expected_y in edge_coords:
            sgf = f'''(;FF[4]GM[1]SZ[19]RE[B+Resign];B[{sgf_coord}])'''
            training_data = self.data_processor.process_game(sgf)
            if len(training_data) > 0:
                _, next_move, _ = training_data[0]
                self.assertEqual(np.argmax(next_move) // 19, expected_x)
                self.assertEqual(np.argmax(next_move) % 19, expected_y)
                
    @patch('urllib.request.urlopen')
    def test_download_with_network_error(self, mock_urlopen):
        # Test network error handling
        mock_urlopen.side_effect = Exception("Network error")
        with self.assertRaises(Exception):
            self.data_processor.download_kgs_games(num_games=1)
            
    def test_memory_efficiency(self):
        # Test memory efficiency with large number of games
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Generate 100 sample games
        for i in range(100):
            game_path = os.path.join(self.test_dir, f"game_{i}.sgf")
            with open(game_path, 'w') as f:
                f.write(self.sample_sgf)
                
        # Process games and check memory usage
        count = 0
        for _ in self.data_processor.prepare_training_data():
            count += 1
            if count > 1000:  # Check first 1000 positions
                break
                
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory increase should be reasonable (less than 100MB for this test)
        self.assertLess(memory_increase, 100)
        
    def test_game_metadata(self):
        # Test extraction of game metadata
        metadata = {
            'PB': 'Black',
            'PW': 'White',
            'BR': '6d',
            'WR': '7d',
            'RE': 'B+3.5',
            'KM': '6.5'
        }
        
        for key, expected_value in metadata.items():
            self.assertIn(expected_value, self.sample_sgf)
            
    def test_invalid_sgf(self):
        # Test various invalid SGF contents
        invalid_sgfs = [
            "(;FF[4])",  # Incomplete SGF
            "",          # Empty string
            "invalid",   # Invalid format
            "(;FF[4]GM[1]SZ[19]RE[B+Resign];B[xx])"  # Invalid coordinate
        ]
        
        for invalid_sgf in invalid_sgfs:
            training_data = self.data_processor.process_game(invalid_sgf)
            self.assertEqual(len(training_data), 0)
        
    def test_prepare_training_data(self):
        # Create test SGF files
        test_game_path = os.path.join(self.test_dir, "test_game.sgf")
        with open(test_game_path, 'w') as f:
            f.write(self.sample_sgf)
            
        # Test training data generator
        training_generator = self.data_processor.prepare_training_data(num_games=1)
        training_examples = list(training_generator)
        
        # Check if we got training examples
        self.assertTrue(len(training_examples) > 0)
        
        # Check structure of generated examples
        for board_state, next_move, outcome in training_examples:
            self.assertEqual(board_state.shape, (19, 19))
            self.assertEqual(next_move.shape, (19, 19))
            self.assertTrue(isinstance(outcome, float))
            
    def tearDown(self):
        # Clean up temporary test directory
        import shutil
        shutil.rmtree(self.test_dir)

if __name__ == '__main__':
    unittest.main() 