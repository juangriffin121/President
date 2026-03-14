• Prioritized improvement plan (including your items + additional ones):

  1. P0 Fix exchange-card path end-to-end
     Importance: Critical
     Why: this is game-defining logic and can currently break for agent players.
     What to do:

      - Table.game() already calls exchange in table.py, so the feature is wired.
      - Real gap: AgentStrategy.choose_worst() delegates to abstract base and returns None in strategy.py, which can fail assert choice is not None in exchange flow.
      - Implement choose_worst for AgentStrategy (or agent class), and add tests for president/scum exchange across multiple games.
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
  8. P3 UX/perf polish
     Importance: Low-Medium
     What to do: cleaner CLI prompts, optional action masking hints, speed optimizations for possible_sets() when hand is large.


› So in linear agent state and hand features dont do anything?


• In your current LinearAgent setup: essentially yes.

  - hand/state features are identical for every action in a decision.
  - In softmax over action scores, any constant added to all actions cancels out.
  - In your gradient update, those shared columns also cancel exactly because sum(one_hot - p) = 0.

  So they neither affect action probabilities nor get meaningful learning signal (apart from tiny numerical noise).
  Only action-varying features are doing real work.


› in the other models they do matter?


• Yes, in MLPAgent and AttentionAgent, state/hand features can matter.

  - LinearAgent: no interactions, so shared state/hand term is just a constant offset across actions and cancels.
  - MLPAgent: because of nonlinear layers (leaky_relu), state can interact with action features and change relative action scores.
  - AttentionAgent: explicitly uses state embedding (L_s) against action embeddings (L_a), so state directly changes action ranking.

  Only exception: if your MLP were purely linear end-to-end (no nonlinearity), it would collapse to the same cancellation behavior as linear.


› they still dont perform too well compared to linear, they are better but not much better and AA learns harder


• That’s expected with your current data flow.

  Main reason: all three are still trained with high-variance REINFORCE and tiny reward signal (-2..2 at game end). Better architectures help, but variance dominates.

  What likely limits gains most:

  1. Credit assignment is too delayed

  - One terminal reward for many turns.
  - Complex models suffer more from noisy gradients.

  2. Action set + policy mismatch

  - Many legal actions, sparse good ones.
  - Softmax over noisy scores makes AA unstable.

  3. Feature scaling / optimization sensitivity

  - AA especially is sensitive to init scale, temp, and LR.
  - Your current random std 4 is very large for dot-product latent models.

  Fastest practical fixes (before more feature work):

  1. Reduce AA init scale drastically

  - Use ~0.05-0.2 std (not 4).

  2. Lower AA learning rate

  - AA should usually learn with smaller dt than linear.

  3. Add entropy bonus decay or temperature schedule tuned per model

  - Prevent early collapse, then sharpen later.

  4. Normalize advantages per game/batch

  - You already tried baseline; per-batch standardization often helps more.

  5. Add intermediate rewards

  - Even simple shaping (e.g., small reward for hand reduction / winning round lead) improves signal-to-noise.

  If you want, I can propose a concrete hyperparameter preset specifically for your current AA code that should make training much more stable without changing algorithm class.
