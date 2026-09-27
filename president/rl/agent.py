from __future__ import annotations

import numpy as np

from president.card import Card, Joker
from president.rl.card_choosers import (
    ActorCriticChooser,
    LinearChooser,
    MLPChooser,
    StateScorerChooser,
)
from president.rl.chooser import CardChooser, WorstChooser
from president.rl.features import Features, get_features
from president.state import GlobalState, PlayerState
from president.rules import valid_choice
from president.utils import possible_sets


class Agent:
    def __init__(self, card_chooser: CardChooser) -> None:
        self.card_chooser = card_chooser
        self.worst_chooser = WorstChooser()
        self.trajectory: list[tuple[Features, int, np.ndarray]] = []
        self.pending_games: list[tuple[list[tuple[Features, int, np.ndarray]], int]] = []
        self.pending_worst: list[tuple[tuple, int]] = []

    def choose_cards(
        self, state: GlobalState, player_state: PlayerState
    ) -> list[Card | Joker] | None:
        last_played = state.played[-1] if state.played else None
        valid_actions = self.get_valid_actions(last_played, player_state.hand)
        features = get_features(state, player_state, valid_actions)
        self.card_chooser.initialize(features)
        probs = self.card_chooser.get_probabilities(features)
        choice_idx, choice = self.choose(valid_actions, probs)
        if not self.card_chooser.frozen:
            self.trajectory.append((features, choice_idx, probs))
        return choice

    def choose_worst(self, count, hand) -> list[Card | Joker]:
        return self.worst_chooser.choose(
            count=count,
            hand=hand,
            temperature=self.card_chooser.temperature,
            frozen=self.card_chooser.frozen,
        )

    def update(self, reward: int) -> None:
        self.card_chooser.update(self.trajectory, reward)
        self.worst_chooser.update(
            reward=reward,
            dt=self.dt,
            temperature=self.temperature,
            frozen=self.frozen,
        )
        self.trajectory = []

    def record_game(self, reward: int) -> None:
        """Stash this game's trajectory for a later batched update instead of applying now."""
        self.pending_games.append((self.trajectory, reward))
        worst_cache = self.worst_chooser.pop_cache()
        if worst_cache is not None:
            self.pending_worst.append((worst_cache, reward))
        self.trajectory = []

    def apply_batch(self) -> float:
        if not self.pending_games:
            return 0.0
        advantage = self.card_chooser.update_batch(self.pending_games)
        self.worst_chooser.update_batch(
            self.pending_worst,
            dt=self.dt,
            temperature=self.temperature,
            frozen=self.frozen,
        )
        self.pending_games = []
        self.pending_worst = []
        return advantage

    def choose(
        self, actions: list[list[Card | Joker] | None], probs: np.ndarray
    ) -> tuple[int, list[Card | Joker] | None]:
        assert len(actions) >= 1
        if len(actions) == 1:
            return (0, actions[0])
        assert probs.size == len((actions))
        total = probs.sum()
        if total <= 0:
            idx = int(np.random.randint(0, len(actions)))
            return (idx, actions[idx])
        probs = probs / total
        idx = int(np.random.choice(len(actions), p=probs))
        return (idx, actions[idx])

    def get_valid_actions(
        self, last_played: list[Card | Joker] | None, hand: list[Card | Joker]
    ) -> list[list[Card | Joker] | None]:
        possible: list[list[Card | Joker] | None] = list(possible_sets(hand))
        if last_played is None:
            return possible
        valid = [choice for choice in possible if valid_choice(choice, last_played)]
        valid.append(None)
        return valid

    def freeze(self):
        self.card_chooser.frozen = True

    def unfreeze(self):
        self.card_chooser.frozen = False

    def save(self, path: str) -> None:
        payload = self.card_chooser.save_payload()
        for key, value in self.worst_chooser.save_payload().items():
            payload[f"worst__{key}"] = value
        np.savez(path, **payload)

    def clone(self, perturb_std: float = 0.1) -> "Agent":
        clone = Agent(self.card_chooser.clone(perturb_std=perturb_std))
        return clone

    @classmethod
    def load(cls, path: str) -> Agent:
        checkpoint = np.load(path, allow_pickle=False)
        kind = str(checkpoint["kind"])
        match kind:
            case "linear":
                chooser = LinearChooser.load(checkpoint)
            case "mlp":
                chooser = MLPChooser.load(checkpoint)
            case "state_scorer":
                chooser = StateScorerChooser.load(checkpoint)
            case "actor_critic":
                chooser = ActorCriticChooser.load(checkpoint)
            case _:
                raise ValueError(f"Unknown agent kind in checkpoint: {kind}")

        agent = cls(chooser)

        key = "worst__num_linear_layers"
        if key in checkpoint:
            num = int(checkpoint[key])
            payload = {"num_linear_layers": checkpoint[key]}
            for i in range(num):
                payload[f"w{i}"] = checkpoint[f"worst__w{i}"]
                payload[f"b{i}"] = checkpoint[f"worst__b{i}"]
            agent.worst_chooser.load_payload(payload)

        return agent

    @property
    def dt(self) -> float:
        return self.card_chooser.dt

    @dt.setter
    def dt(self, value: float) -> None:
        self.card_chooser.dt = value

    @property
    def temperature(self) -> float:
        return self.card_chooser.temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        self.card_chooser.temperature = value

    @property
    def frozen(self) -> bool:
        return self.card_chooser.frozen

    @frozen.setter
    def frozen(self, value: bool) -> None:
        self.card_chooser.frozen = value
