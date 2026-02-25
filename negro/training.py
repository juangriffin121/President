from dataclasses import dataclass, field
from typing import Callable

from player import Player, set_sleep_enabled
from rl_numpy import Agent
from strategy import AgentStrategy, Smallest
from table import Table
from ui import writes

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TrainingLog:
    rewards: list[int] = field(default_factory=list)
    extras: list[dict] = field(default_factory=list)


def train(
    num_games: int,
    agent: Agent | None = None,
    log_hook: Callable[[int, AgentStrategy, Table], dict] | None = None,
) -> TrainingLog:
    writes.set_silent(True)
    set_sleep_enabled(False)

    if agent is None:
        agent = Agent()

    agent_strategy = AgentStrategy(agent)
    p1 = Player("p1", Smallest())
    p2 = Player("p2", Smallest())
    a = Player("agent", agent_strategy)

    log = TrainingLog()
    for game_idx in range(num_games):
        if game_idx > 0 and game_idx % 200 == 0:
            agent.temperature = max(0.3, agent.temperature * 0.80)
            agent.dt = max(0.05, agent.dt * 0.90)

        t = Table([p1, p2, a])
        t.game()
        log.rewards.append(agent_strategy.last_reward or 0)

        print(f"{game_idx} dt: {agent.dt}, temp: {agent.temperature}")
        print(agent_strategy.last_reward)

        if log_hook is not None:
            log.extras.append(log_hook(game_idx, agent_strategy, t))
    return log


log = train(3000)

rewards = np.array(log.rewards, dtype=float)
window = 50
if len(rewards) >= window:
    moving = np.convolve(rewards, np.ones(window) / window, mode="valid")
else:
    moving = rewards

plt.figure(figsize=(8, 4))
plt.plot(rewards, alpha=0.35, label="reward")
if len(moving) > 0:
    plt.plot(
        range(window - 1, window - 1 + len(moving)), moving, label=f"avg({window})"
    )
for y in [-2, -1, 0, 1, 2]:
    plt.axhline(y, color="gray", linewidth=0.8, alpha=0.4)
plt.xlabel("game")
plt.ylabel("reward")
plt.legend()
plt.tight_layout()
plt.show()
