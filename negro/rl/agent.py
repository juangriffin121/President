from abc import ABC, abstractmethod
from numpy import ndarray
import numpy as np
from negro.card import Card, Joker
from negro.state import GlobalState, PlayerState
from negro.rules import valid_choice
from negro.utils import possible_sets
from negro.ranking import get_num, order_num


class Agent(ABC):
    def __init__(self) -> None:
        self.trajectory = []
        self.worst_chosen: list[Card | Joker] = []
        self.frozen = False
        self.dt = 0.1  # Redefined in subclasses
        self.temperature = 1.0  # Redefined in subclasses
        self.reward_count = 0
        self.reward_mean = 0.0
        self.reward_m2 = 0.0
        self.adv_clip = 3.0

    def choose_cards(
        self, state: GlobalState, player_state
    ) -> list[Card | Joker] | None:
        last_played = state.played[-1] if state.played else None
        valid_actions = self.get_valid_actions(last_played, player_state.hand)
        features = self.get_features(state, player_state, valid_actions)
        self.init_weights(features.shape[1])
        probs = self.get_probabilities(features)
        choice_idx, choice = self.choose(valid_actions, probs)
        if not self.frozen:
            self.trajectory.append((features, choice_idx, probs))
        return choice

    @abstractmethod
    def init_weights(self, num_features) -> None:
        raise NotImplementedError

    def choose_worst(self, count, hand):
        # code
        if not self.frozen:
            pass
            # self.worst_chosen =

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(self, reward: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_probabilities(self, features: ndarray) -> ndarray:
        raise NotImplementedError

    def choose(
        self, actions: list[list[Card | Joker] | None], probs: ndarray
    ) -> tuple[int | None, list[Card | Joker] | None]:
        if len(actions) == 0:
            return (None, None)
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
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def _normalize_reward(self, reward: float) -> float:
        if self.reward_count < 2:
            normalized = reward
        else:
            variance = self.reward_m2 / (self.reward_count - 1)
            std = max(float(np.sqrt(variance)), 1e-6)
            normalized = (reward - self.reward_mean) / std

        normalized = float(np.clip(normalized, -self.adv_clip, self.adv_clip))
        self._update_reward_stats(reward)
        return normalized

    def _update_reward_stats(self, reward: float) -> None:
        self.reward_count += 1
        delta = reward - self.reward_mean
        self.reward_mean += delta / self.reward_count
        delta2 = reward - self.reward_mean
        self.reward_m2 += delta * delta2

    def get_features(
        self,
        state: GlobalState,
        player_state: PlayerState,
        valid_actions: list[list[Card | Joker] | None],
    ) -> ndarray:
        hand = player_state.hand
        player_id = player_state.id
        hand_count = len(hand)
        initial_hand = 50 / state.total_players
        safe_hand_count = max(1, hand_count)
        safe_other_total = max(1, 50 - initial_hand)

        jokers_in_hand = sum(1 for c in hand if isinstance(c, Joker))
        rank_counts = [0] * 12
        for c in hand:
            if isinstance(c, Card):
                rank_counts[c.num - 1] += 1

        num_distinct_ranks = sum(1 for cnt in rank_counts if cnt > 0)
        num_pairs = sum(1 for cnt in rank_counts if cnt >= 2)
        num_triples = sum(1 for cnt in rank_counts if cnt >= 3)
        num_quads = sum(1 for cnt in rank_counts if cnt >= 4)

        last_play_exists = 1 if state.played else 0
        last_play = state.played[-1] if state.played else None
        last_play_rank = get_num(last_play) if last_play else 0
        last_play_rank_norm = order_num(last_play_rank) / 13 if last_play else 0
        last_play_count = len(last_play) if last_play else 0

        players_left = state.total_players - len(state.winners)
        winners = len(state.winners)

        is_presidente = (
            int(player_id == state.presidente) if state.presidente is not None else 0
        )
        is_vice_presidente = (
            int(player_id == state.vice_presidente)
            if state.vice_presidente is not None
            else 0
        )
        is_vice_negro = (
            int(player_id == state.vice_negro) if state.vice_negro is not None else 0
        )
        is_negro = int(player_id == state.negro) if state.negro is not None else 0

        bin_cards = state.bin
        bin_count = len(bin_cards)
        bin_jokers = sum(1 for c in bin_cards if isinstance(c, Joker))
        bin_aces = sum(1 for c in bin_cards if isinstance(c, Card) and c.num == 1)
        if bin_count > 0:
            bin_avg_rank_norm = (
                sum(order_num(c.num) for c in bin_cards if isinstance(c, Card))
                / max(1, bin_count - bin_jokers)
            ) / 13
        else:
            bin_avg_rank_norm = 0

        sum_other_hand_sizes = sum(state.hand_sizes) - hand_count
        avg_other_hand_size = (
            sum_other_hand_sizes / max(1, players_left - 1) if players_left > 1 else 0
        )

        state_features = [
            hand_count / initial_hand,
            jokers_in_hand / 2,
            *[cnt / 4 for cnt in rank_counts],
            num_distinct_ranks / 12,
            num_pairs / (initial_hand / 2),
            num_triples / (initial_hand / 3),
            num_quads / (initial_hand / 4),
            last_play_exists,
            last_play_rank_norm,
            last_play_count / 6,
            players_left / state.total_players,
            winners / state.total_players,
            is_presidente,
            is_vice_presidente,
            is_vice_negro,
            is_negro,
            bin_count / 50,
            bin_jokers / 2,
            bin_aces / 4,
            bin_avg_rank_norm,
            sum_other_hand_sizes / safe_other_total,
            avg_other_hand_size / initial_hand,
        ]

        feature_rows = []
        for action in valid_actions:
            action_is_pass = 1 if action is None else 0
            action_count = 0 if action is None else len(action)
            action_rank = 0 if action is None else order_num(get_num(action))
            action_rank_norm = action_rank / 13 if action is not None else 0
            jokers_used = (
                sum(1 for c in action if isinstance(c, Joker)) if action else 0
            )
            uses_ace = (
                1
                if action is not None
                and any(isinstance(c, Card) and c.num == 1 for c in action)
                else 0
            )
            empties_hand = 1 if action and action_count == hand_count else 0
            fraction_hand_used = (
                action_count / safe_hand_count if action is not None else 0
            )
            beats_last_by = (
                (action_rank - order_num(last_play_rank)) / 13
                if action is not None and last_play
                else 0
            )

            action_features = [
                action_is_pass,
                action_count / 6,
                action_rank_norm,
                jokers_used / 2,
                uses_ace,
                empties_hand,
                fraction_hand_used,
                beats_last_by,
            ]

            feature_rows.append(state_features + action_features)

        return np.array(feature_rows, dtype=float)


class LinearAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.weights: ndarray | None = None
        self.dt = 0.6
        self.temperature = 3.0

    def init_weights(self, num_features) -> None:
        if self.weights is None:
            self.weights = np.random.normal(0.0, 4, size=num_features)

    def update(self, reward: int) -> None:
        assert not self.frozen
        advantage = self._normalize_reward(float(reward))
        for features, choice_idx, probs in self.trajectory:
            self.update_weights(features, choice_idx, probs, advantage)
        self.trajectory = []

    def update_weights(
        self, features: ndarray, choice_idx: int, probs: ndarray, reward: float
    ) -> None:
        assert self.weights is not None
        grad = softmax_grad(probs, self.temperature, choice_idx, reward)
        self.weights += self.dt * grad @ features

    def get_probabilities(self, features: ndarray) -> ndarray:
        assert self.weights is not None
        if features.size == 0:  # No possible actions
            return np.zeros(features.shape[0], dtype=float)
        scores = features @ self.weights
        return softmax(scores, self.temperature)

    def save(self, path: str) -> None:
        assert self.weights is not None
        np.savez(
            path,
            version=np.array(1, dtype=int),
            kind=np.array("linear"),
            dt=np.array(self.dt, dtype=float),
            temperature=np.array(self.temperature, dtype=float),
            frozen=np.array(int(self.frozen), dtype=int),
            weights=self.weights,
        )


class MLPAgent(Agent):
    def __init__(self, hidden_layers_sizes: tuple[int, ...]) -> None:
        super().__init__()

        self.dt = 0.2
        self.temperature = 2.0

        self.hidden_layers_sizes = hidden_layers_sizes
        self.weights: list[ndarray] | None = None
        self.biases: list[ndarray] | None = None
        self.neuron_log: list[tuple[list[ndarray], list[ndarray]]] = []

    def init_weights(self, num_features) -> None:
        if self.weights is None:
            # layers_shape contains hidden layer sizes only.
            layer_sizes = (num_features, *self.hidden_layers_sizes, 1)
            self.weights = []
            self.biases = []
            for layer_size, next_layer_size in zip(layer_sizes, layer_sizes[1:]):
                self.weights.append(
                    np.random.normal(0.0, 0.1, size=(layer_size, next_layer_size))
                )
                self.biases.append(np.random.normal(0.0, 0.1, size=(next_layer_size)))

    def update(self, reward: int) -> None:
        assert not self.frozen
        advantage = self._normalize_reward(float(reward))
        for (_, choice_idx, probs), (x_cache, y_cache) in zip(
            self.trajectory, self.neuron_log
        ):
            self.update_weights(choice_idx, probs, advantage, x_cache, y_cache)
        self.trajectory = []
        self.neuron_log = []

    def update_weights(
        self,
        choice_idx: int,
        probs: ndarray,
        reward: float,
        x_cache: list[ndarray],
        y_cache: list[ndarray],
    ) -> None:
        assert self.weights is not None
        assert self.biases is not None

        dy = softmax_grad(probs, self.temperature, choice_idx, reward)[:, None]

        for i in reversed(range(len(self.weights))):
            w = self.weights[i]
            x = x_cache[i]

            dW = x.T @ dy
            db = dy.sum(axis=0)  # b is shape (1,)

            if i > 0:
                dX = dy @ w.T
                y = y_cache[i - 1]
                dy = dX * leaky_relu_grad(y)

            self.weights[i] += self.dt * dW
            self.biases[i] += self.dt * db

    def get_probabilities(self, features: ndarray) -> ndarray:
        assert self.weights is not None
        assert self.biases is not None
        if features.size == 0:  # No possible actions
            return np.zeros(features.shape[0], dtype=float)

        x = features  # shape is (A, F)
        last_idx = len(self.weights) - 1
        x_cache = []
        y_cache = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x_cache.append(x)
            y = x @ w + b  # w shape is (H-, H+) b:(H+,) (hidden) x:(A, H-) y:(A, H+)
            y_cache.append(y)
            if i == last_idx:
                x = y
            else:
                x = leaky_relu(y)

        if not self.frozen:
            self.neuron_log.append((x_cache, y_cache))

        x = x.squeeze(1)

        return softmax(x, self.temperature)

    def save(self, path: str) -> None:
        assert self.weights is not None
        assert self.biases is not None
        payload: dict[str, ndarray] = {
            "version": np.array(1, dtype=int),
            "kind": np.array("mlp"),
            "dt": np.array(self.dt, dtype=float),
            "temperature": np.array(self.temperature, dtype=float),
            "frozen": np.array(int(self.frozen), dtype=int),
            "hidden_layers_sizes": np.array(self.hidden_layers_sizes, dtype=int),
            "num_layers": np.array(len(self.weights), dtype=int),
        }
        for idx, w in enumerate(self.weights):
            payload[f"w{idx}"] = w
        for idx, b in enumerate(self.biases):
            payload[f"b{idx}"] = b
        np.savez(path, **payload)


def load_agent(path: str) -> Agent:
    checkpoint = np.load(path, allow_pickle=False)
    kind = str(checkpoint["kind"])
    if kind == "linear":
        agent = LinearAgent()
        agent.weights = checkpoint["weights"]
    elif kind == "mlp":
        hidden_layers = tuple(
            int(x) for x in checkpoint["hidden_layers_sizes"].tolist()
        )
        agent = MLPAgent(hidden_layers)
        num_layers = int(checkpoint["num_layers"])
        agent.weights = [checkpoint[f"w{i}"] for i in range(num_layers)]
        agent.biases = [checkpoint[f"b{i}"] for i in range(num_layers)]
    else:
        raise ValueError(f"Unknown agent kind in checkpoint: {kind}")

    agent.dt = float(checkpoint["dt"])
    agent.temperature = float(checkpoint["temperature"])
    agent.frozen = bool(int(checkpoint["frozen"]))
    agent.trajectory = []
    agent.worst_chosen = []
    if isinstance(agent, MLPAgent):
        agent.neuron_log = []
    return agent


def clone_agent(agent: Agent, perturb_std: float = 0.0) -> Agent:
    if isinstance(agent, LinearAgent):
        clone = LinearAgent()
        assert agent.weights is not None
        clone.weights = agent.weights.copy()
        if perturb_std > 0:
            clone.weights += np.random.normal(0.0, perturb_std, size=clone.weights.shape)
    elif isinstance(agent, MLPAgent):
        clone = MLPAgent(agent.hidden_layers_sizes)
        assert agent.weights is not None
        assert agent.biases is not None
        clone.weights = [w.copy() for w in agent.weights]
        clone.biases = [b.copy() for b in agent.biases]
        if perturb_std > 0:
            clone.weights = [
                w + np.random.normal(0.0, perturb_std, size=w.shape)
                for w in clone.weights
            ]
            clone.biases = [
                b + np.random.normal(0.0, perturb_std, size=b.shape)
                for b in clone.biases
            ]
    else:
        raise ValueError(f"Unsupported agent type for cloning: {type(agent).__name__}")

    clone.dt = agent.dt
    clone.temperature = agent.temperature
    clone.frozen = agent.frozen
    clone.reward_count = agent.reward_count
    clone.reward_mean = agent.reward_mean
    clone.reward_m2 = agent.reward_m2
    clone.adv_clip = agent.adv_clip
    clone.trajectory = []
    clone.worst_chosen = []
    return clone


def leaky_relu(x: ndarray) -> ndarray:
    return np.where(x > 0, x, 0.1 * x)


def leaky_relu_grad(y: ndarray) -> ndarray:
    return np.where(y > 0, 1, 0.1)


def softmax(x: ndarray, temp: float) -> ndarray:
    x = x / max(temp, 1e-6)
    max_x = np.max(x) if x.size else 0
    exp_x = np.exp(x - max_x)
    total = exp_x.sum()
    if total <= 0:
        return np.zeros_like(exp_x)
    return exp_x / total


def softmax_grad(y: ndarray, temp: float, choice_idx: int, reward: float) -> ndarray:
    one_hot = np.zeros(y.shape[0])
    one_hot[choice_idx] = 1
    return reward * (one_hot - y) / max(temp, 1e-6)
