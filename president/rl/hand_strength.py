import numpy as np

from president.card import Card, Joker
from president.rl.features import get_hand_features


class HandStrengthPredictor:
    def __init__(self, dt: float = 0.05, l2: float = 0) -> None:
        self.dt = dt
        self.l2 = l2
        self.w: np.ndarray | None = None
        self._last_features: np.ndarray | None = None
        self.frozen: bool = False

    def observe_hand(self, hand: list[Card | Joker], total_players: int) -> None:
        x = np.array(get_hand_features(hand, total_players))
        if self.w is None:
            self.w = np.zeros_like(x)
        self._last_features = x

    def predict_from_features(self, x: np.ndarray) -> float:
        if self.w is None:
            self.w = np.zeros_like(x)
        y = float(x @ self.w)
        return float(2*np.tanh(y))

    def predict_hand(self, hand: list[Card | Joker], total_players: int) -> float:
        return self.predict_from_features(
            np.array(get_hand_features(hand, total_players))
        )

    def update(self, actual_reward: int) -> float | None:
        if self._last_features is None:
            return None
        x = self._last_features
        assert self.w is not None
        y = float(x @ self.w)
        pred = float(2 * np.tanh(y))

        if not self.frozen:
            err = pred - float(actual_reward)
            grad_factor = err * 2 * (1 - np.tanh(y) ** 2)
            self.w -= self.dt * (grad_factor * x + self.l2 * self.w)
        return pred

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False
