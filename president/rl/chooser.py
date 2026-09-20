from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

from president.card import Card, Joker
from president.nn.layers import Linear
from president.nn.network import NeuralNetwork
from president.rl.features import NUM_CARD_FEATS, Features, get_card_features


@dataclass
class CardChooser(ABC):
    def __init__(self) -> None:
        self.dt = 0.1
        self.temperature = 1.0
        self.frozen = False

    @abstractmethod
    def initialize(self, features: Features) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_probabilities(self, features: Features) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def update(
        self, trajectory: list[tuple[Features, int, np.ndarray]], reward: int
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def save_payload(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def clone(self, perturb_std: float = 0.1) -> "CardChooser":
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, checkpoint) -> "CardChooser":
        raise NotImplementedError

    @abstractmethod
    def update_batch(
        self, games: list[tuple[list[tuple[Features, int, np.ndarray]], int]]
    ) -> float:
        """Apply one averaged update across several (trajectory, reward) games."""
        raise NotImplementedError

# currently only used for WorstChooser
class RunningBaseline:
    """Running mean that turns into an EMA once enough samples have been seen."""

    def __init__(self, lr: float = 0.05) -> None:
        self.lr = lr
        self.value = 0.0
        self.n = 0

    def advantage(self, reward: float) -> float:
        # No estimate yet: give no learning signal instead of comparing against a fake 0.
        return reward - self.value if self.n > 0 else 0.0

    def update(self, reward: float) -> None:
        self.n += 1
        step = max(1.0 / self.n, self.lr)  # 1/n during warm-up, then a plain EMA
        self.value += step * (reward - self.value)

class WorstChooser:
    def __init__(self) -> None:
        self.nn: NeuralNetwork | None = None
        self._cache: tuple[list[int], list[np.ndarray], list[np.ndarray]] = (
            [],
            [],
            [],
        )
        self.baselines: dict[int, RunningBaseline] = {}

    def initialize(self) -> None:
        if self.nn is not None:
            return
        self.nn = NeuralNetwork(NUM_CARD_FEATS, [Linear(1)])
        self.nn.initialize()

    def _baseline(self, num_cards: int) -> RunningBaseline:
        if num_cards not in self.baselines:
            self.baselines[num_cards] = RunningBaseline()
        return self.baselines[num_cards]

    def choose(
        self,
        count: int,
        hand: list[Card | Joker],
        temperature: float,
        frozen: bool,
    ) -> list[Card | Joker]:
        self.initialize()
        assert self.nn is not None

        card_feats = get_card_features(hand)  # (C, Fc)
        scores, cache = self.nn.forward(card_feats.T)  # (Fc, C) -> (1, C)
        probs = _softmax(scores.T[:, 0], temperature)
        idx1 = int(np.random.choice(len(probs), p=probs))
        cards = [hand[idx1]]

        probabilities = [probs]

        idx2 = None
        if count > 1:
            probs2 = probs.copy()
            probs2[idx1] = 0.0
            total = probs2.sum()
            if total > 0:
                probs2 = probs2 / total
            else:
                probs2 = np.zeros_like(probs2)
                probs2[[i for i in range(len(probs2)) if i != idx1]] = 1.0
                probs2 = probs2 / probs2.sum()
            idx2 = int(np.random.choice(len(probs2), p=probs2))
            cards.append(hand[idx2])

            probabilities.append(probs2)

        if not frozen:
            worst_chosen = [idx1]
            if idx2 is not None:
                worst_chosen.append(idx2)
            self._cache = (worst_chosen, probabilities, cache)

        return cards

    def pop_cache(self) -> tuple[list[int], list[np.ndarray], list[np.ndarray]] | None:
        """Return and clear the cache from the last choose() call, or None if it wasn't called."""
        cache = self._cache
        self._cache = ([], [], [])
        return cache if cache[0] else None
    def update_batch(
        self,
        entries: list[tuple[tuple[list[int], list[np.ndarray], list[np.ndarray]], int]],
        dt: float,
        temperature: float,
        frozen: bool,
    ) -> None:
        if frozen or not entries or self.nn is None:
            return
        n = len(entries)

        # Advantages use the baselines as they were BEFORE this batch.
        advantages = [
            self._baseline(len(worst_chosen)).advantage(reward)
            for (worst_chosen, _, _), reward in entries
        ]

        for ((worst_chosen, probabilities, cache), _), advantage in zip(entries, advantages):
            for choice_idx, probs in zip(worst_chosen, probabilities):
                grad = _softmax_grad(probs, temperature, choice_idx, advantage)
                grad_output = -grad[None, :]
                self.nn.backward(grad_output, dt / n, cache)

        for (worst_chosen, _, _), reward in entries:
            self._baseline(len(worst_chosen)).update(reward)

    def update(self, reward: int, dt: float, temperature: float, frozen: bool) -> None:
        cache = self.pop_cache()  # always clears the cache
        if frozen or self.nn is None or cache is None:
            return
        self.update_batch([(cache, reward)], dt=dt, temperature=temperature, frozen=frozen)


    def save_payload(self) -> dict[str, np.ndarray]:
        if self.nn is None:
            return {}
        return self.nn.save_payload()

    def load_payload(self, payload: dict[str, np.ndarray]) -> None:
        if "num_linear_layers" not in payload:
            return
        self.initialize()
        assert self.nn is not None
        self.nn.load(payload)


def _softmax(x: np.ndarray, temp: float) -> np.ndarray:
    x = x / max(temp, 1e-6)
    max_x = np.max(x) if x.size else 0
    exp_x = np.exp(x - max_x)
    total = exp_x.sum()
    if total <= 0:
        return np.zeros_like(exp_x)
    return exp_x / total


def _softmax_grad(
    y: np.ndarray, temp: float, choice_idx: int, reward: float
) -> np.ndarray:
    one_hot = np.zeros(y.shape[0])
    one_hot[choice_idx] = 1
    return reward * (one_hot - y) / max(temp, 1e-6)
