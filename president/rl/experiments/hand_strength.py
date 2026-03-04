from dataclasses import dataclass, field
from typing import Callable

from president.player import Player, set_sleep_enabled
from president.rl.agent import Agent, LinearAgent, MLPAgent
from president.rl.hand_strength import HandStrengthPredictor
from president.strategy import AgentStrategy, Smallest
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
    p1 = Player("p1", Smallest())
    p2 = Player("p2", Smallest())
    a = Player("agent", agent_strategy)

    log = TrainingLog()
    starting_dt = agent.dt
    starting_temp = agent.temperature
    for game_idx in range(num_games):
        if game_idx > 0 and game_idx % 200 == 0:
            agent.temperature = max(0.25 * starting_temp, agent.temperature * 0.90)
            agent.dt = max(0.1 * starting_dt, agent.dt * 0.90)

        t = Table([p1, p2, a])
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


def plot_results(
    rewards: np.ndarray,
    hand_predictions: np.ndarray | None = None,
    agent_name: str = "",
    hand_strength_predictor_name: str = "",
):
    window = 50
    if len(rewards) >= window:
        moving = np.convolve(rewards, np.ones(window) / window, mode="valid")
    else:
        moving = rewards

    fig, (ax_rewards, ax_predictor) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 7), gridspec_kw={"height_ratios": [2, 1]}
    )
    ax_rewards.plot(rewards, alpha=0.35, label="reward")
    if len(moving) > 0:
        ax_rewards.plot(
            range(window - 1, window - 1 + len(moving)),
            moving,
            label=f"avg({window}) {agent_name}",
        )
    for y in [-2, -1, 0, 1, 2]:
        ax_rewards.axhline(
            y,
            color="gray",
            linewidth=0.8,
            alpha=0.4,
        )
    ax_rewards.set_ylabel("reward")
    ax_rewards.legend()

    if hand_predictions is not None and len(hand_predictions) > 0:
        if len(hand_predictions) >= window:
            pred_moving = np.convolve(
                hand_predictions, np.ones(window) / window, mode="valid"
            )
        else:
            pred_moving = hand_predictions
        if len(pred_moving) > 0:
            ax_predictor.plot(
                range(window - 1, window - 1 + len(pred_moving)),
                pred_moving,
                color="tab:blue",
                linewidth=2.0,
                label=f"prediction avg({window}) {hand_strength_predictor_name}",
            )

        if len(moving) > 0:
            ax_predictor.plot(
                range(window - 1, window - 1 + len(moving)),
                moving,
                label=f"actual reward avg({window}) {agent_name}",
                color="tab:orange",
            )
    for y in [-2, -1, 0, 1, 2]:
        ax_predictor.axhline(y, color="gray", linewidth=0.8, alpha=0.25)
    ax_predictor.set_ylim(-2.2, 2.2)
    ax_predictor.set_yticks([-2, -1, 0, 1, 2])
    ax_predictor.legend()
    ax_predictor.set_ylabel("reward")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    mlp_strategy, mlp_log = train(3000)
    mlp = mlp_strategy.agent
    mlp_hand_strength_predictor = mlp_strategy.hand_strength_predictor

    rewards = np.array(mlp_log.rewards, dtype=float)
    hand_strengths = np.array(mlp_log.hand_predictions, dtype=float)
    plot_results(
        rewards,
        hand_strengths,
        agent_name="mlp_ag",
        hand_strength_predictor_name="mlp_hsp",
    )

    l_strategy, l_log = train(3000, LinearAgent())
    l = l_strategy.agent
    l_hand_strength_predictor = l_strategy.hand_strength_predictor

    rewards = np.array(l_log.rewards, dtype=float)
    hand_strengths = np.array(l_log.hand_predictions, dtype=float)
    plot_results(
        rewards,
        hand_strengths,
        agent_name="l_ag",
        hand_strength_predictor_name="l_hsp",
    )

    mlp.freeze()
    l.freeze()
    assert l_hand_strength_predictor is not None
    l_hand_strength_predictor.freeze()
    assert mlp_hand_strength_predictor is not None
    mlp_hand_strength_predictor.freeze()

    mlpl_strategy, mlpl_log = train(1000, mlp, l_hand_strength_predictor)
    mlp = mlpl_strategy.agent
    l_hand_strength_predictor = mlpl_strategy.hand_strength_predictor

    rewards = np.array(mlpl_log.rewards, dtype=float)
    hand_strengths = np.array(mlpl_log.hand_predictions, dtype=float)
    plot_results(
        rewards,
        hand_strengths,
        agent_name="mlp_ag",
        hand_strength_predictor_name="l_hsp",
    )

    lmlp_strategy, lmlp_log = train(1000, l, mlp_hand_strength_predictor)
    l = lmlp_strategy.agent
    mlp_hand_strength_predictor = lmlp_strategy.hand_strength_predictor

    rewards = np.array(lmlp_log.rewards, dtype=float)
    hand_strengths = np.array(lmlp_log.hand_predictions, dtype=float)
    plot_results(
        rewards,
        hand_strengths,
        agent_name="l_ag",
        hand_strength_predictor_name="mlp_hsp",
    )

    assert l_hand_strength_predictor is not None
    assert mlp_hand_strength_predictor is not None
    print(l_hand_strength_predictor.w)
    print(mlp_hand_strength_predictor.w)
