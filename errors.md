• Findings:

  1. valid_choice() accepts illegal mixed-rank leads. In president/rules.py:5, if last_choice is None: return True bypasses all_cards_same(choice), so a lead like [3, 4] is treated as valid. I
     verified valid_choice([Card(3,...), Card(4,...)], None) returns True. That makes the rules inconsistent with possible_sets() and allows UserStrategy to play invalid openings.
  2. play.py and rl/fight.py both import a non-existent load_agent, so those entry points fail immediately on import. See president/play.py:2 and president/rl/fight.py:3 versus president/rl/
     agent.py:97, which only defines Agent.load(...). I verified from president.rl.agent import load_agent raises ImportError.
  4. Agent.clone() drops the learned worst_chooser state. president/rl/agent.py:93 only clones card_chooser, even though save()/load() preserve worst_chooser weights at president/rl/agent.py:87
     and president/rl/agent.py:115. That matters because the population experiment replaces agents via best_agent.clone(...) in president/rl/experiments/agent_family.py:187, so exchange-card
     behavior gets reset on every replacement.
