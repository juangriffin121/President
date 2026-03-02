from typing import Callable
from negro.player import Player, set_sleep_enabled
from negro.rl.agent import Agent, load_agent
from negro.strategy import AgentStrategy, Smallest
from negro.table import Table
from negro.ui import writes


def fight(
    num_games: int,
    agents: list[Agent],
):
    writes.set_silent(True)
    set_sleep_enabled(False)

    players = []
    for i, agent in enumerate(agents):
        agent.freeze()
        agent_strategy = AgentStrategy(agent)
        players.append(Player("agent", agent_strategy))

    for game_idx in range(num_games):
        t = Table(players)
        t.game()
