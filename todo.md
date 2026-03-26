  Target architecture

  - Agent becomes a thin orchestrator composed of:
      - CardChooser (abstract interface): initialize, get_probabilities, update, save, load, clone.
      - WorstChooser (keeps existing behavior, but implemented as another chooser).
  - Each current agent type becomes a CardChooser implementation rather than a subclass of Agent.
      - LinearAgent → LinearCardChooser
      - MLPAgent → MLPCardChooser
      - StateScorerAgent → StateScorerCardChooser
      - ActorCritic → ActorCriticCardChooser
  - Agent stays responsible for:
      - computing valid_actions
      - calling CardChooser.get_probabilities
      - action sampling
      - trajectory storage
      - calling CardChooser.update
      - coordinating WorstChooser

  Unified run/experiment utilities

  - Create a single runner module, e.g. president/rl/runner.py, with:
      - run_games(...) (shared train/test loop)
      - build_players(...)
      - TrainingLog
      - plot_results(...)
      - set_silent_training_mode(...)
  - Update president/rl/train.py, president/rl/test.py, and experiment files to use the runner.

  Additional touch points (low-risk)

  - Move math helpers softmax, softmax_grad, leaky_relu from president/rl/agent.py into a shared module like president/rl/ops.py or president/nn/ops.py.
  - Consider making president/rl/experiments/* thin “scripts” that call the runner instead of duplicating logic.
  - Optional: unify seeds and randomness via a set_seed helper in president/rl/runner.py (currently only in president/rl/experiments/agent_family.py).

  ———

  Proposed refactor plan

  1. Introduce the new chooser interfaces and keep behavior identical.
      - Add CardChooser base class in president/rl/chooser.py (or president/rl/choosers/base.py).
      - Add WorstChooser implementation that wraps current worst-chooser NN logic.
      - Adapt Agent to use a CardChooser instance instead of being abstract.
  2. Split current agent subclasses into chooser implementations.
      - Move LinearAgent, MLPAgent, StateScorerAgent, ActorCritic logic into chooser classes.
      - Keep load_agent as a factory that returns an Agent with the right CardChooser.
  3. Extract runner utilities and de-duplicate training/testing.
      - Create president/rl/runner.py with shared run_games, TrainingLog, plot_results.
      - Simplify president/rl/train.py, president/rl/test.py, president/rl/experiments/*.py to use it.
  4. Cleanup pass.
      - Move math ops into president/rl/ops.py.
      - Update imports and remove dead code paths.
