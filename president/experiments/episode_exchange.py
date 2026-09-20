from dataclasses import dataclass, field

from president.player import Player, set_sleep_enabled
from president.rl.agent import Agent
from president.rl.card_choosers import (
    ActorCriticChooser,
    LinearChooser,
    MLPChooser,
    StateScorerChooser,
)
from president.rl.train import plot_results, train
from president.strategy import AgentStrategy, Smallest
from president.table import Table
from president.ui import writes
import numpy as np


@dataclass
class TrainingLog:
    rewards: list[int] = field(default_factory=list)


def train_with_exchange(
    num_episodes: int,
    games_per_episode: int = 5,
    agent: Agent | None = None,
    batch_size: int | None = None,
) -> tuple[AgentStrategy, TrainingLog]:
    writes.set_silent(True)
    set_sleep_enabled(False)

    if agent is None:
        agent = Agent(LinearChooser())

    agent_strategy = AgentStrategy(agent, batch_size=batch_size or games_per_episode)
    log = TrainingLog()

    for episode_idx in range(num_episodes):
        opponents = [
            Player("p1", Smallest()),
            Player("p2", Smallest()),
            Player("p3", Smallest()),
        ]
        agent_player = Player("agent", agent_strategy)
        table = Table([*opponents, agent_player])

        for game_idx in range(games_per_episode):
            table.game()
            reward = agent_strategy.last_reward or 0
            log.rewards.append(reward)

        print(
            f"episode={episode_idx + 1}/{num_episodes} "
            f"games_per_episode={games_per_episode}"
        )

    if agent_strategy.agent.pending_games:
        agent_strategy.agent.apply_batch()

    return agent_strategy, log


if __name__ == "__main__":

    # agent = Agent(LinearChooser(dt=batch_size*0.6, temperature=3))  
    agent = Agent(StateScorerChooser(dt=2, temperature=5))  
    # agent = Agent(MLPChooser((128, 32, 8)))
    """
    strategy, log = train(3000, agent, batch_size=5)
    rewards = np.array(log.rewards, dtype=float)
    plot_results(
       rewards,
       agent_name="Agent",
    )
    agent = strategy.agent
    """

    strategy, log = train_with_exchange(
        num_episodes=500, games_per_episode=20, agent=agent, batch_size=5
    )
    rewards = np.array(log.rewards, dtype=float)
    plot_results(
        rewards,
        agent_name="Agent",
    )
