from typing import Callable
from president.player import Player, set_sleep_enabled
from president.rl.agent import Agent, load_agent
from president.rl.train import TrainingLog, plot_results
from president.strategy import AgentStrategy, Smallest
from president.table import Table
from president.ui import writes

import numpy as np


def test(
    num_games: int,
    agent: Agent,
    log_hook: Callable[[int, AgentStrategy, Table], dict] | None = None,
) -> TrainingLog:
    writes.set_silent(True)
    set_sleep_enabled(False)

    agent.freeze()
    agent_strategy = AgentStrategy(agent)
    p1 = Player("p1", Smallest())
    p2 = Player("p2", Smallest())
    p3 = Player("p3", Smallest())
    p4 = Player("p4", Smallest())
    a = Player("agent", agent_strategy)

    log = TrainingLog()
    for game_idx in range(num_games):
        t = Table([p1, p2, a])
        t.game()
        log.rewards.append(agent_strategy.last_reward or 0)

        print(f"{game_idx} dt: {agent.dt}, temp: {agent.temperature}")
        print(agent_strategy.last_reward)

        if log_hook is not None:
            log.extras.append(log_hook(game_idx, agent_strategy, t))
    return log


agent = load_agent("Linear.npz")
log = test(1000, agent)
rewards = np.array(log.rewards, dtype=float)
avg_reward = np.average(rewards)
plot_results(rewards)
print(f"Average: {avg_reward}")
