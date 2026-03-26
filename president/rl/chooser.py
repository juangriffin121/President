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


class WorstChooser:
    def __init__(self) -> None:
        self.nn: NeuralNetwork | None = None
        self._cache: tuple[list[int], list[np.ndarray], list[np.ndarray]] = (
            [],
            [],
            [],
        )

    def initialize(self) -> None:
        if self.nn is not None:
            return
        self.nn = NeuralNetwork(NUM_CARD_FEATS, [Linear(1)])
        self.nn.initialize()

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

    def update(
        self, advantage: float, dt: float, temperature: float, frozen: bool
    ) -> None:
        if frozen:
            self._cache = ([], [], [])
            return
        if self.nn is None:
            self._cache = ([], [], [])
            return
        worst_chosen, probabilities, cache = self._cache
        if not worst_chosen:
            return
        for choice_idx, probs in zip(worst_chosen, probabilities):
            grad = _softmax_grad(probs, temperature, choice_idx, advantage)
            grad_output = grad[None, :]  # (1, C)
            self.nn.backward(grad_output, dt, cache)
        self._cache = ([], [], [])

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
