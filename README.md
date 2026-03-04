# President

Python implementation of the card game **President** with:

- Core game engine (deck, rules, rounds, role exchange)
- Multiple strategies (human input, heuristic bots, RL agent strategy)
- Reinforcement learning agents implemented in NumPy (linear + MLP)
- Training and experiment scripts

## Requirements

- Python `3.11+`
- [Poetry](https://python-poetry.org/)

## Setup

```bash
poetry install
```

## How To Run

Run commands from the project root.

### 1) Play a game (human + bots)

```bash
poetry run python -m president.play
```

This starts an interactive terminal game with:

- 2 heuristic bots
- 1 pretrained MLP agent (`MLPA_7_3.npz`)
- You (manual input)

### 2) Train an RL agent

```bash
poetry run python -m president.rl.train
```

Runs training and shows reward plots.

### 3) Evaluate a saved agent

Edit the checkpoint path inside `president/rl/test.py` (default is `Linear.npz`), then run:

```bash
poetry run python -m president.rl.test
```

### 4) Agent-vs-agent fights

Use `president/rl/fight.py` as an entry point for multi-agent matches.

## Run Tests

```bash
poetry run pytest -q
```

## Project Structure

```text
president/
  card.py            # Card and Joker types
  deck.py            # Deck construction + shuffle/deal data
  rules.py           # Move validation rules
  ranking.py         # Card/order helpers
  table.py           # Round/game loop and role transitions
  player.py          # Player state + strategy hooks
  strategy.py        # Human/bot/agent strategies
  play.py            # Interactive game entry point
  ui/                # Terminal read/write helpers
  rl/
    agent.py         # Linear/MLP policy agents + save/load
    features.py      # State/action feature extraction
    hand_strength.py # Hand-strength predictor
    train.py         # Training/evaluation utilities
    test.py          # Evaluation script
    fight.py         # Agent-vs-agent battles
    experiments/     # Experiment scripts
tests/               # Pytest suite
```

## Notes

- The game uses terminal input/output for interactive play.
- Pretrained `.npz` checkpoints in the repo root can be loaded by RL scripts.
- If imports fail, prefer running through Poetry from the repository root (commands above).
