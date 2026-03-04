import numpy as np

from negro.card import Card, Joker
from negro.rl.features import get_hand_features


class HandStrengthPredictor:
    def __init__(self, dt: float = 0.05, l2: float = 1e-4) -> None:
        self.dt = dt
        self.l2 = l2
        self.w: np.ndarray | None = None
        self.b = 0.0
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
        y = float(x @ self.w + self.b)
        return float(np.clip(y, -2.0, 2.0))

    def predict_hand(self, hand: list[Card | Joker], total_players: int) -> float:
        return self.predict_from_features(
            np.array(get_hand_features(hand, total_players))
        )

    def update(self, actual_reward: int) -> float | None:
        if self._last_features is None:
            return None
        x = self._last_features
        pred = self.predict_from_features(x)

        if not self.frozen:
            err = pred - float(actual_reward)
            assert self.w is not None
            self.w -= self.dt * (err * x + self.l2 * self.w)
            self.b -= self.dt * err
        return pred

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False
