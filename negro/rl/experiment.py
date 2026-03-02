from dataclasses import dataclass, field
from pathlib import Path
import random
from typing import Callable
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt

from negro.player import Player, set_sleep_enabled
from negro.rl.agent import Agent, LinearAgent, MLPAgent, clone_agent
from negro.strategy import AgentStrategy, Smallest
from negro.table import Table
from negro.ui import writes


@dataclass
class ExperimentConfig:
    num_agents: int = (
        3  # How many agents will be trained in parallel, high for better statistics, low for faster evaluation
    )
    checkpoint_interval: int = (
        500  # The interval between checkpoints, a checkpoint is a point in the training step where the agents are tested and the best one saved
    )
    max_games: int = 2000
    test_games_per_checkpoint: int = 200
    num_opponents: int = 3
    train_base_seed: int = (
        1000  # base seed for training games, actual seed for each games is computed from this number, to ensure replicability and make games fair for agents
    )
    eval_base_seed: int = (
        9000  # base seed for testing games, actual seed for each game is computed from this number, to ensure replicability and make games fair for agents
    )
    output_dir: str = "artifacts"
    anneal_every_games: int = 200  # Num games before annealing, deminishing dt and temp
    min_temperature_pct: float = 0.25
    min_dt_pct: float = 0.1
    temp_annealing_coef: float = 0.9
    dt_annealing_coef: float = 0.9
    replacement_per_checkpoint: int = 1
    replacement_noise_std: float = 0.02
    colors: list[str] = field(
        default_factory=lambda: ["blue", "red", "green", "cyan", "magenta", "yellow"]
    )


@dataclass
class FamilyResult:
    train_rewards: np.ndarray
    test_rewards: np.ndarray
    temp_10pct_games: np.ndarray
    dt_10pct_games: np.ndarray


def run_family_experiment(
    agent_factory: Callable[[], Agent], config: ExperimentConfig
) -> FamilyResult:
    writes.set_silent(True)
    set_sleep_enabled(False)

    out = Path(config.output_dir)
    agents_dir = out / "agents"
    plots_dir = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    agents_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = list(
        range(
            config.checkpoint_interval, config.max_games + 1, config.checkpoint_interval
        )
    )
    if not checkpoints:
        raise ValueError(
            "No checkpoints generated; verify checkpoint_interval and max_games"
        )

    agents = [agent_factory() for _ in range(config.num_agents)]
    init_temps = np.array([agent.temperature for agent in agents], dtype=float)
    init_dts = np.array([agent.dt for agent in agents], dtype=float)
    temp_10pct_games = np.full(config.num_agents, -1, dtype=int)
    dt_10pct_games = np.full(config.num_agents, -1, dtype=int)
    games_trained = [
        0 for _ in range(config.num_agents)
    ]  # How many games each agent played at that point
    train_rewards = [
        [] for _ in range(config.num_agents)
    ]  # reward per training game per agent
    test_rewards = [
        [] for _ in range(config.num_agents)
    ]  # reward per testing game per agent
    eval_seeds = [
        config.eval_base_seed + i for i in range(config.test_games_per_checkpoint)
    ]
    train_log_every = max(1, config.checkpoint_interval // 5)

    family_name = _agent_base_name(agents[0]) if agents else "Unknown"
    print(
        f"[START] family={family_name} agents={config.num_agents} "
        f"checkpoints={len(checkpoints)} max_games={config.max_games}"
    )

    for checkpoint in checkpoints:
        print(f"[CHECKPOINT] family={family_name} target_games={checkpoint}")
        test_means = []
        for agent_id, agent in enumerate(agents):
            print(
                f"[TRAIN] family={family_name} agent={agent_id} "
                f"start={games_trained[agent_id]} target={checkpoint}"
            )
            # until agent reaches checkpoint train agent
            while games_trained[agent_id] < checkpoint:
                game_num = games_trained[agent_id]

                # annealing
                if game_num > 0 and game_num % config.anneal_every_games == 0:
                    min_temp = init_temps[agent_id] * config.min_temperature_pct
                    min_dt = init_dts[agent_id] * config.min_dt_pct
                    agent.temperature = max(
                        min_temp, agent.temperature * config.temp_annealing_coef
                    )
                    agent.dt = max(min_dt, agent.dt * config.dt_annealing_coef)
                    if (
                        temp_10pct_games[agent_id] == -1
                        and agent.temperature <= 0.1 * init_temps[agent_id]
                    ):
                        temp_10pct_games[agent_id] = games_trained[agent_id]
                    if (
                        dt_10pct_games[agent_id] == -1
                        and agent.dt <= 0.1 * init_dts[agent_id]
                    ):
                        dt_10pct_games[agent_id] = games_trained[agent_id]

                train_seed = config.train_base_seed + agent_id * 1_000_000 + game_num
                reward = _play_one_game(
                    agent,
                    num_opponents=config.num_opponents,
                    seed=train_seed,
                )
                train_rewards[agent_id].append(reward)
                games_trained[agent_id] += 1
                if (
                    games_trained[agent_id] % train_log_every == 0
                    or games_trained[agent_id] == checkpoint
                ):
                    print(
                        f"[TRAIN] family={family_name} agent={agent_id} "
                        f"games={games_trained[agent_id]}/{checkpoint}"
                    )

            print(
                f"[TEST] family={family_name} agent={agent_id} "
                f"games={games_trained[agent_id]} eval_games={config.test_games_per_checkpoint}"
            )
            test_reward = _evaluate_agent(
                agent,
                config.test_games_per_checkpoint,
                config.num_opponents,
                eval_seeds,
            )
            test_rewards[agent_id].append(test_reward)

            mean_test_reward = float(np.mean(test_reward))
            test_means.append(mean_test_reward)
            print(
                f"[TEST] family={family_name} agent={agent_id} "
                f"mean={mean_test_reward:.4f}"
            )

        best_agent_id = int(np.argmax(np.array(test_means, dtype=float)))
        best_agent = agents[best_agent_id]
        print(
            f"[CHECKPOINT_DONE] family={family_name} target_games={checkpoint} "
            f"best_agent={best_agent_id} best_mean={test_means[best_agent_id]:.4f}"
        )

        base_name = _agent_base_name(best_agent)
        ckpt_label = _checkpoint_label(checkpoint)
        best_name = f"{base_name}__{ckpt_label}__S__{config.num_opponents}.npz"
        best_agent.save(str(agents_dir / best_name))

        if config.replacement_per_checkpoint > 0 and config.num_agents > 1:
            ordered = np.argsort(np.array(test_means, dtype=float))
            worst_ids = [int(i) for i in ordered if int(i) != best_agent_id]
            replace_count = min(config.replacement_per_checkpoint, len(worst_ids))
            for replaced_id in worst_ids[:replace_count]:
                agents[replaced_id] = clone_agent(
                    best_agent, perturb_std=config.replacement_noise_std
                )
                print(
                    f"[POP] family={family_name} checkpoint={checkpoint} "
                    f"replaced_agent={replaced_id} source_best={best_agent_id} "
                    f"noise_std={config.replacement_noise_std}"
                )

    return FamilyResult(
        train_rewards=np.array(train_rewards),
        test_rewards=np.array(test_rewards),
        temp_10pct_games=temp_10pct_games,
        dt_10pct_games=dt_10pct_games,
    )


def run_experiment(
    families: dict[str, Callable[[], Agent]],
    config: ExperimentConfig,
) -> dict[str, FamilyResult]:
    if config.num_agents <= 0:
        raise ValueError("num_runs must be > 0")
    if config.test_games_per_checkpoint <= 0:
        raise ValueError("test_games_per_checkpoint must be > 0")
    if config.checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be > 0")
    if config.max_games <= 0:
        raise ValueError("max_games must be > 0")
    if config.max_games % config.checkpoint_interval != 0:
        raise ValueError("max_games must be divisible by checkpoint_interval")
    if config.num_opponents <= 0:
        raise ValueError("num_opponents must be > 0")

    results: dict[str, FamilyResult] = {}
    for family_name, agent_factory in families.items():
        results[family_name] = run_family_experiment(
            agent_factory=agent_factory,
            config=config,
        )

    plot(results, config)

    return results


def plot(
    experiment_results: dict[str, FamilyResult],
    config: ExperimentConfig,
):
    plots_dir = Path(config.output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = np.arange(
        config.checkpoint_interval, config.max_games + 1, config.checkpoint_interval
    )

    fig, (ax_spaghetti, ax_test) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(11, 8),
        gridspec_kw={"height_ratios": [2, 1]},
    )

    for idx, (family_name, family_result) in enumerate(experiment_results.items()):
        color = config.colors[idx % len(config.colors)]
        plot_family_spaghetti(
            family_result.train_rewards,
            family_result.temp_10pct_games,
            family_result.dt_10pct_games,
            ax_spaghetti,
            color,
            family_name,
        )
        plot_family_test_results(
            family_result.test_rewards, checkpoints, ax_test, color, family_name
        )

    for y in [-2, -1, 0, 1, 2]:
        ax_spaghetti.axhline(y, color="gray", linewidth=0.8, alpha=0.4)
        ax_test.axhline(y, color="gray", linewidth=0.8, alpha=0.4)

    for ckpt in checkpoints:
        ax_spaghetti.axvline(
            ckpt, color="gray", linewidth=0.7, alpha=0.25, linestyle="--"
        )
        ax_test.axvline(ckpt, color="gray", linewidth=0.7, alpha=0.25, linestyle="--")

    ax_spaghetti.set_title("Training Spaghetti + Checkpoint Test Performance")
    ax_spaghetti.set_ylabel("train reward (moving avg)")
    ax_test.set_xlabel("games trained")
    ax_test.set_ylabel("test reward")

    ax_spaghetti.legend()
    ax_test.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "spaghetti_curves.png", dpi=160)
    plt.close()


def plot_family_spaghetti(
    rewards: np.ndarray,
    temp_10pct_games: np.ndarray,
    dt_10pct_games: np.ndarray,
    ax: Axes,
    color: str,
    family_name: str,
) -> None:
    window = 50
    curves = [_get_curve(agent_rewards, window) for agent_rewards in rewards]

    for curve in curves:
        if len(curve) > 0:
            ax.plot(
                range(window - 1, window - 1 + len(curve)),
                curve,
                alpha=0.18,
                color=color,
            )

    for x in temp_10pct_games:
        if x >= 0:
            ax.axvline(x, color=color, linestyle=":", linewidth=1.2, alpha=0.7)
            ax.text(
                x,
                0.98,
                "temp",
                color=color,
                fontsize=7,
                rotation=90,
                va="top",
                ha="right",
                transform=ax.get_xaxis_transform(),
                alpha=0.95,
            )

    for x in dt_10pct_games:
        if x >= 0:
            ax.axvline(x, color=color, linestyle=":", linewidth=1.2, alpha=0.7)
            ax.text(
                x,
                0.90,
                "dt",
                color=color,
                fontsize=7,
                rotation=90,
                va="top",
                ha="right",
                transform=ax.get_xaxis_transform(),
                alpha=0.95,
            )

    mean_curve = np.mean(np.array(curves), axis=0)
    ax.plot(
        range(window - 1, window - 1 + len(mean_curve)),
        mean_curve,
        color=color,
        linewidth=2.2,
        label=family_name,
    )


def plot_family_test_results(
    test_results: np.ndarray,
    checkpoints: np.ndarray,
    ax: Axes,
    color: str,
    family_name: str,
) -> None:
    # test_results shape: (num_agents, num_checkpoints, num_eval_games)
    run_checkpoint_means = test_results.mean(axis=2)
    num_agents = run_checkpoint_means.shape[0]

    # scatter for all runs at each checkpoint
    x_scatter = np.tile(checkpoints, num_agents)
    y_scatter = run_checkpoint_means.reshape(-1)
    ax.scatter(x_scatter, y_scatter, color=color, alpha=0.22, s=16)

    # family mean line over checkpoints
    family_mean = run_checkpoint_means.mean(axis=0)
    ax.plot(checkpoints, family_mean, color=color, linewidth=2.0, label=family_name)


def _get_curve(rewards: np.ndarray, window: int) -> np.ndarray:
    if len(rewards) >= window:
        moving = np.convolve(rewards, np.ones(window) / window, mode="valid")
    else:
        moving = rewards
    return moving


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))


def _build_table(agent_strategy: AgentStrategy, num_opponents: int) -> Table:
    players: list[Player] = []
    for i in range(num_opponents):
        players.append(Player(f"opp{i + 1}", Smallest()))
    players.append(Player("agent", agent_strategy))
    return Table(players)


def _play_one_game(
    agent: Agent,
    num_opponents: int,
    seed: int,
) -> int:
    _set_seed(seed)
    strategy = AgentStrategy(agent)
    table = _build_table(strategy, num_opponents)
    table.game()
    return strategy.last_reward or 0


def _evaluate_agent(
    agent: Agent,
    num_games: int,
    num_opponents: int,
    eval_seeds: list[int],
) -> np.ndarray:
    was_frozen = agent.frozen
    agent.freeze()
    rewards = []
    for i in range(num_games):
        seed = eval_seeds[i]
        reward = _play_one_game(agent, num_opponents, seed)
        rewards.append(reward)
    if not was_frozen:
        agent.unfreeze()
    return np.array(rewards, dtype=float)


def _agent_base_name(agent: Agent) -> str:
    if isinstance(agent, LinearAgent):
        return "L"

    hidden = getattr(agent, "hidden_layers_sizes", None)
    if hidden is not None:
        txt = "_".join(str(int(x)) for x in hidden)
        return f"MLP__{txt}" if txt else "MLP"

    return type(agent).__name__


def _checkpoint_label(games_trained: int) -> str:
    k = games_trained / 1000
    if float(k).is_integer():
        return f"{int(k)}k"
    return f"{k:g}k"


if __name__ == "__main__":
    # One-family smoke test config (edit as needed before running).
    families: dict[str, Callable[[], Agent]] = {
        # "Linear": lambda: LinearAgent(),
        "MLP5": lambda: MLPAgent((5,)),
        "MLP7-3": lambda: MLPAgent((7, 3)),
        "MLP20": lambda: MLPAgent((20,)),
    }

    config = ExperimentConfig(
        num_agents=5,
        checkpoint_interval=100,
        max_games=2000,
        test_games_per_checkpoint=100,
        num_opponents=3,
        output_dir="artifacts_test_two_families",
        anneal_every_games=100,
    )

    run_experiment(families, config)
