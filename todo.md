• Prioritized improvement plan (including your items + additional ones):

  1. P0 Fix exchange-card path end-to-end
     Importance: Critical
     Why: this is game-defining logic and can currently break for agent players.
     What to do:

      - Table.game() already calls exchange in table.py, so the feature is wired.
      - Real gap: AgentStrategy.choose_worst() delegates to abstract base and returns None in strategy.py, which can fail assert choice is not None in exchange flow.
      - Implement choose_worst for AgentStrategy (or agent class), and add tests for president/scum exchange across multiple games.

  2. P0 Add tests for core rules/game progression
     Importance: Critical
     Why: many asserts enforce game legality; regressions will be silent until runtime.
     What to cover: valid_choice, joker behavior, round reset/pass rules, finishing order, role assignment, exchange correctness.
  3. P1 Refactor repeated training/testing code
     Importance: High
     Why: easier experimentation, fewer bugs.
     What to do: unify duplicated loops in rl/train.py, rl/test.py, and experiments/* into shared helpers (run_episode, evaluate_agent, build_players).
  4. P1 Add baseline strategies (Random, GreedyBig, ConservativePass)
     Importance: High
     Why: better baselines make RL progress measurable.
     What to do: implement in strategy.py, then benchmark with fixed seeds.
  5. P1 Improve “fight” evaluation into real benchmarking
     Importance: High
     Why: current fight() is minimal and doesn’t return metrics.
     What to add: mean reward, rank distribution, win rate, confidence intervals, Elo-like ratings, per-opponent matrix.
  6. P2 Module naming/structure cleanup
     Importance: Medium
     Why: improves discoverability and maintenance.
     What to do: split into core/ (cards/rules/table), agents/, strategies/, sim/, ui/; rename play.py to cli_play.py, fight.py to benchmark.py.
  7. P2 Determinism and reproducibility
     Importance: Medium
     Why: essential for comparing learning changes.
     What to do: central seed manager and explicit seed logging for train/eval runs.
  8. P2 Type safety and static checks
     Importance: Medium
     Why: many Card | Joker | None paths are fragile.
     What to do: tighten signatures, run mypy/ruff, replace broad except: in round flow with specific handling.
  9. P3 UX/perf polish
     Importance: Low-Medium
     What to do: cleaner CLI prompts, optional action masking hints, speed optimizations for possible_sets() when hand is large.
