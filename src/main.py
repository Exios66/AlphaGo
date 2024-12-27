#!/usr/bin/env python3

import argparse
from ui.game_ui import GameUI

def main():
    parser = argparse.ArgumentParser(description='AlphaGo Clone - A Go game with AI')
    parser.add_argument('--board-size', type=int, default=19,
                      help='Size of the Go board (default: 19)')
    parser.add_argument('--ai-mode', action='store_true',
                      help='Enable AI opponent (not implemented yet)')
    
    args = parser.parse_args()
    
    if args.board_size not in [9, 13, 19]:
        print("Warning: Non-standard board size. Using 19x19 board.")
        args.board_size = 19
    
    print(f"Starting AlphaGo Clone with {args.board_size}x{args.board_size} board")
    if args.ai_mode:
        print("AI opponent mode is not implemented yet. Starting in two-player mode.")
    
    game = GameUI(board_size=args.board_size)
    game.run()

if __name__ == "__main__":
    main() 