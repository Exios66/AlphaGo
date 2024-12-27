# AlphaGo Clone

An AlphaGo-inspired application that allows users to play the game of Go against an AI opponent. This project demonstrates the integration of deep learning models with Monte Carlo Tree Search to create a competitive Go AI.

## Features

- Interactive Go Board: Play on customizable 9x9, 13x13, or 19x19 grids with real-time updates
- AI Opponent: Powered by Policy and Value Networks combined with Monte Carlo Tree Search
- Move Validation: Ensures all moves comply with Go rules including ko and suicide rules
- Game Visualization: Display captured stones, territory estimation, and game status
- Training Pipeline: Scripts to train neural networks on game datasets
- Pre-trained Models: Ready-to-use policy and value networks trained on KGS game records
- Cross-platform Support: Runs on Windows, macOS, and Linux

## Installation

### Prerequisites

- Python 3.8+
- PyTorch >= 1.9.0
- Pygame >= 2.0.1
- NumPy >= 1.21.0
- Matplotlib >= 3.4.3
- tqdm >= 4.62.3
- pytest >= 6.2.5

### Setup

1. Clone the repository:
