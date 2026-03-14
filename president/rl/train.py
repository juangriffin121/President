from dataclasses import dataclass, field
from typing import Callable

from president.player import Player, set_sleep_enabled
from president.rl.agent import Agent, LinearAgent, MLPAgent, StateScorerAgent
from president.rl.features import action_feat_names, hand_feat_names, state_feat_names
from president.rl.hand_strength import HandStrengthPredictor
from president.strategy import AgentStrategy, Random, Smallest
from president.table import Table
from president.ui import writes

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TrainingLog:
    rewards: list[int] = field(default_factory=list)
    hand_predictions: list[float] = field(default_factory=list)
    predictor_right: list[int] = field(default_factory=list)
    extras: list[dict] = field(default_factory=list)


def train(
    num_games: int,
    agent: Agent | None = None,
    hand_strength_predictor: HandStrengthPredictor | None = None,
    log_hook: Callable[[int, AgentStrategy, Table], dict] | None = None,
    players: list[Player] | None = None,
) -> tuple[AgentStrategy, TrainingLog]:
    writes.set_silent(True)
    set_sleep_enabled(False)

    if agent is None:
        agent = MLPAgent((20,))  # LinearAgent()

    if hand_strength_predictor is None:
        hand_strength_predictor = HandStrengthPredictor()

    agent_strategy = AgentStrategy(
        agent, hand_strength_predictor=hand_strength_predictor
    )
    if players is None:
        p1 = Player("p1", Smallest())
        p2 = Player("p2", Smallest())
        p3 = Player("p3", Smallest())
        p4 = Player("p4", Smallest())
        players = [p1, p2, p3, p4]
    a = Player("agent", agent_strategy)

    log = TrainingLog()
    starting_dt = agent.dt
    starting_temp = agent.temperature
    for game_idx in range(num_games):
        if game_idx > 0 and game_idx % 200 == 0:
            agent.temperature = max(0.25 * starting_temp, agent.temperature * 0.90)
            agent.dt = max(0.1 * starting_dt, agent.dt * 0.90)

        t = Table([*players, a])
        t.game()
        reward = agent_strategy.last_reward or 0
        prediction = agent_strategy.last_hand_strength_prediction
        if prediction is None:
            prediction = 0.0
        predicted_class = int(np.clip(np.rint(prediction), -2, 2))
        log.rewards.append(reward)
        log.hand_predictions.append(float(prediction))
        log.predictor_right.append(int(predicted_class == reward))

        print(f"{game_idx} dt: {agent.dt}, temp: {agent.temperature}")
        print(agent_strategy.last_reward)

        if log_hook is not None:
            log.extras.append(log_hook(game_idx, agent_strategy, t))

    print(hand_strength_predictor.w)
    print(hand_strength_predictor.b)
    return agent_strategy, log


def test(
    num_games: int,
    agent: Agent,
    log_hook: Callable[[int, AgentStrategy, Table], dict] | None = None,
    players: list[Player] | None = None,
) -> TrainingLog:
    writes.set_silent(True)
    set_sleep_enabled(False)

    agent.freeze()
    hand_strength_predictor = HandStrengthPredictor()
    agent_strategy = AgentStrategy(
        agent, hand_strength_predictor=hand_strength_predictor
    )
    if players is None:
        p1 = Player("p1", Smallest())
        p2 = Player("p2", Smallest())
        p3 = Player("p3", Smallest())
        p4 = Player("p4", Smallest())
        players = [p1, p2, p3, p4]
    a = Player("agent", agent_strategy)

    log = TrainingLog()
    for game_idx in range(num_games):
        t = Table([*players, a])
        t.game()
        reward = agent_strategy.last_reward or 0
        prediction = agent_strategy.last_hand_strength_prediction
        if prediction is None:
            prediction = 0.0
        predicted_class = int(np.clip(np.rint(prediction), -2, 2))
        log.rewards.append(reward)
        log.hand_predictions.append(float(prediction))
        log.predictor_right.append(int(predicted_class == reward))

        print(f"{game_idx} dt: {agent.dt}, temp: {agent.temperature}")
        print(agent_strategy.last_reward)

        if log_hook is not None:
            log.extras.append(log_hook(game_idx, agent_strategy, t))
    return log


def plot_results(
    rewards: np.ndarray,
    agent_name: str = "",
):
    window = 50
    if len(rewards) >= window:
        moving = np.convolve(rewards, np.ones(window) / window, mode="valid")
    else:
        moving = rewards

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(rewards, alpha=0.35, label="reward")
    if len(moving) > 0:
        ax.plot(
            range(window - 1, window - 1 + len(moving)),
            moving,
            label=f"avg({window}) {agent_name}",
        )
    for y in [-2, -1, 0, 1, 2]:
        ax.axhline(
            y,
            color="gray",
            linewidth=0.8,
            alpha=0.4,
        )
    ax.set_ylabel("reward")
    ax.legend()

    for y in [-2, -1, 0, 1, 2]:
        ax.axhline(y, color="gray", linewidth=0.8, alpha=0.25)
    ax.legend()
    ax.set_ylabel("reward")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    agent = StateScorerAgent()  # MLPAgent((128, 32, 8))
    strategy, log = train(2000, agent)
    agent = strategy.agent
    players = [
        Player("p1", Smallest()),
        Player("p2", Smallest()),
        Player("p3", Random()),
    ]

    rewards = np.array(log.rewards, dtype=float)
    plot_results(
        rewards,
        agent_name="Agent",
    )

    log = test(2000, agent)
    rewards = np.array(log.rewards, dtype=float)
    plot_results(
        rewards,
        agent_name="Agent",
    )

    players = [
        Player("p1", Random()),
        Player("p2", Random()),
        Player("p3", Random()),
    ]
    log = test(2000, agent, players=players)
    rewards = np.array(log.rewards, dtype=float)
    plot_results(
        rewards,
        agent_name="Agent",
    )

    feature_names = hand_feat_names() + state_feat_names() + action_feat_names()
    assert isinstance(agent, LinearAgent)
    weights = agent.weights
    assert weights is not None
    assert len(feature_names) == weights.size

    for name, weight in zip(feature_names, weights.squeeze()):
        print(f"{name}: {weight:.2f}")
