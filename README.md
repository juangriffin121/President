# President

Python implementation of the card game **President** with:

- Core game engine (deck, rules, rounds, role exchange)
- Multiple strategies (human input, heuristic bots, RL agent strategy)
- Reinforcement learning agents implemented in NumPy (linear + MLP)
- Training scripts and optimization experiments

Game reference: [President (card game) on Wikipedia](https://en.wikipedia.org/wiki/President_(card_game)).

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



## specifics


  For refactor brainstorming (no code changes), I’d use experiments/agent_family.py as the canonical base and split like this:

  Shared reusable core

  - play_one_game(agent, num_opponents, seed, hand_strength_predictor=None) -> game_result
  - evaluate_agent(agent, seeds, num_opponents) -> np.ndarray
  - anneal(agent, initial_dt, initial_temp, game_idx, config) -> None
  - set_reproducible_seed(seed) -> None
  - common TrainingLog/EvalLog dataclasses
  - optional callback hooks: on_game_end, on_checkpoint_end

  Belongs to train.py (single-agent workflow)

  - default agent creation
  - simple loop over num_games
  - optional annealing schedule
  - optional hand-strength predictor updates/logging
  - lightweight plotting/CLI behavior for quick iteration

  Belongs to test.py (evaluation-only workflow)

  - load checkpoint
  - freeze agent
  - run fixed number of eval games
  - report summary stats (mean reward, maybe std/quantiles)

  Belongs to experiments/agent_family.py (population workflow)

  - multiple agents in parallel
  - checkpoint cadence
  - best-agent selection on eval seeds
  - replacement/cloning of weak agents
  - artifact management (save checkpoints + experiment plots)

  A clean boundary is:

  - core module has zero knowledge of “families/checkpoints/replacement”
  - experiment module composes core primitives to implement those policies

  If you want, next step I can draft a concrete target module layout and function signatures only (no edits yet), so we can agree before refactoring.
