from typing import Callable
from president.player import Player, set_sleep_enabled
from president.rl.agent import Agent, load_agent
from president.strategy import AgentStrategy, Smallest
from president.table import Table
from president.ui import writes


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
