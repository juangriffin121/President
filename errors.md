• Findings:

  1. valid_choice() accepts illegal mixed-rank leads. In president/rules.py:5, if last_choice is None: return True bypasses all_cards_same(choice), so a lead like [3, 4] is treated as valid. I
     verified valid_choice([Card(3,...), Card(4,...)], None) returns True. That makes the rules inconsistent with possible_sets() and allows UserStrategy to play invalid openings.
  2. play.py and rl/fight.py both import a non-existent load_agent, so those entry points fail immediately on import. See president/play.py:2 and president/rl/fight.py:3 versus president/rl/
     agent.py:97, which only defines Agent.load(...). I verified from president.rl.agent import load_agent raises ImportError.
  3. Table.deal() calls player.on_deal() after every single card instead of once after the full hand is dealt, and before card exchange happens. See president/table.py:28 and president/
     table.py:162, plus the consumer in president/strategy.py:216 and predictor state in president/rl/hand_strength.py:16. I verified observe_hand() is called 25 times for a 2-player deal. On
     later games this means the predictor is trained on the pre-exchange hand, not the actual starting hand after president/scum swaps.
  4. Agent.clone() drops the learned worst_chooser state. president/rl/agent.py:93 only clones card_chooser, even though save()/load() preserve worst_chooser weights at president/rl/agent.py:87
     and president/rl/agent.py:115. That matters because the population experiment replaces agents via best_agent.clone(...) in president/rl/experiments/agent_family.py:187, so exchange-card
     behavior gets reset on every replacement.
