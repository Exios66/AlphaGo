# AlphaGo Clone

An AlphaGo-inspired application that allows users to play the game of Go against an AI opponent. This project demonstrates the integration of deep learning models with Monte Carlo Tree Search to create a competitive Go AI.

## Features

- Interactive Go Board: Play on a 19x19 grid with real-time updates
- AI Opponent: Powered by Policy and Value Networks combined with Monte Carlo Tree Search
- Move Validation: Ensures all moves comply with Go rules
- Game Visualization: Display captured stones and game status

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- Pygame
- Other dependencies listed in `requirements.txt`

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/AlphaGo-Clone.git
cd AlphaGo-Clone
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
python src/main.py
```

## Project Structure

```
AlphaGo-Clone/
├── data/
│   ├── datasets/          # Go game datasets
│   └── models/            # Pre-trained models
├── src/
│   ├── game/             # Game logic and rules
│   ├── ai/               # AI components (Policy Network, Value Network, MCTS)
│   ├── ui/               # User interface components
│   └── utils/            # Utility functions
├── notebooks/            # Jupyter notebooks for experimentation
├── tests/                # Unit and integration tests
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## License

This project is licensed under the MIT License.

## Acknowledgements

- Inspired by DeepMind's AlphaGo
- Utilizes PyTorch for deep learning
- Game data sourced from KGS Go Dataset
