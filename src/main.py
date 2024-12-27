#!/usr/bin/env python3

import argparse
from ui.game_ui import GameUI

def main():
    parser = argparse.ArgumentParser(description='AlphaGo Clone - A Go game with AI')
    parser.add_argument('--board-size', type=int, default=19,
                      help='Size of the Go board (default: 19)')
    parser.add_argument('--ai-mode', action='store_true',
                      help='Enable AI opponent (plays as White)')
    parser.add_argument('--num-simulations', type=int, default=800,
                      help='Number of MCTS simulations per move (default: 800)')
    
    args = parser.parse_args()
    
    if args.board_size not in [9, 13, 19]:
        print("Warning: Non-standard board size. Using 19x19 board.")
        args.board_size = 19
    
    print(f"Starting AlphaGo Clone with {args.board_size}x{args.board_size} board")
    if args.ai_mode:
        print("AI opponent enabled (playing as White)")
        print(f"Using {args.num_simulations} MCTS simulations per move")
    
    game = GameUI(board_size=args.board_size, ai_opponent=args.ai_mode)
    game.run()

if __name__ == "__main__":
    main() 