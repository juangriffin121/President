from abc import ABC, abstractmethod
from numpy import ndarray, number
import numpy as np
from president.card import Card, Joker
from president.rl.features import Features, get_features
from president.state import GlobalState, PlayerState
from president.rules import valid_choice
from president.utils import possible_sets
from president.ranking import get_num, order_num


class Agent(ABC):
    def __init__(self) -> None:
        self.trajectory: list[tuple[Features, int, ndarray]] = []
        self.worst_chosen: list[Card | Joker] = []
        self.frozen = False
        self.dt = 0.1  # Redefined in subclasses
        self.temperature = 1.0  # Redefined in subclasses

    def choose_cards(
        self, state: GlobalState, player_state
    ) -> list[Card | Joker] | None:
        last_played = state.played[-1] if state.played else None
        valid_actions = self.get_valid_actions(last_played, player_state.hand)
        features = get_features(state, player_state, valid_actions)
        self.init_weights(features)
        probs = self.get_probabilities(features)
        choice_idx, choice = self.choose(valid_actions, probs)
        if not self.frozen:
            self.trajectory.append((features, choice_idx, probs))
        return choice

    @abstractmethod
    def init_weights(self, features: Features) -> None:
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
    def get_probabilities(self, features: Features) -> ndarray:
        raise NotImplementedError

    def choose(
        self, actions: list[list[Card | Joker] | None], probs: ndarray
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
        self.frozen = True

    def unfreeze(self):
        self.frozen = False


class LinearAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.weights: ndarray | None = None
        self.dt = 0.6
        self.temperature = 3.0

    def init_weights(self, features: Features) -> None:
        if self.weights is None:
            num_features = features.state_hand.size + features.actions.shape[1]
            self.weights = np.random.normal(0.0, 4, size=num_features)

    def update(self, reward: int) -> None:
        assert not self.frozen
        for features, choice_idx, probs in self.trajectory:
            self.update_weights(features, choice_idx, probs, reward)
        self.trajectory = []

    def update_weights(
        self, features: Features, choice_idx: int, probs: ndarray, reward: float
    ) -> None:
        assert self.weights is not None
        flat_features = features.as_concatenated()
        grad = softmax_grad(probs, self.temperature, choice_idx, reward)
        self.weights += self.dt * grad @ flat_features

    def get_probabilities(self, features: Features) -> ndarray:
        flat_features = features.as_concatenated()
        assert self.weights is not None
        if flat_features.size == 0:  # No possible actions
            return np.zeros(flat_features.shape[0], dtype=float)
        scores = flat_features @ self.weights
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

    def init_weights(self, features: Features) -> None:
        if self.weights is None:
            # layers_shape contains hidden layer sizes only.
            num_features = features.state_hand.size + features.actions.shape[1]
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
        for (_, choice_idx, probs), (x_cache, y_cache) in zip(
            self.trajectory, self.neuron_log
        ):
            self.update_weights(choice_idx, probs, reward, x_cache, y_cache)
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

        dR_dy = softmax_grad(probs, self.temperature, choice_idx, reward)[:, None]

        for i in reversed(range(len(self.weights))):
            w = self.weights[i]
            x = x_cache[i]

            dR_dW = (
                x.T @ dR_dy
            )  # y = x @ W -> dR_dW = dR_dy @ dy_dW, dy_dW = x: (A, Fs) dR_dy: (A, )
            dR_db = dR_dy.sum(axis=0)  # b is shape (1,) dR_dy is shape (A, )

            if i > 0:
                dR_dX = dR_dy @ w.T
                y = y_cache[i - 1]
                dR_dy = dR_dX * leaky_relu_grad(y)

            self.weights[i] += self.dt * dR_dW
            self.biases[i] += self.dt * dR_db

    def get_probabilities(self, features: Features) -> ndarray:
        assert self.weights is not None
        assert self.biases is not None
        flat_features = features.as_concatenated()
        if flat_features.size == 0:  # No possible actions
            return np.zeros(flat_features.shape[0], dtype=float)

        x = flat_features  # shape is (A, F)
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


class AttentionAgent(Agent):
    def __init__(self, latent_size: int = 10) -> None:
        super().__init__()
        self.ws: ndarray | None = None
        self.bs: ndarray | None = None
        self.wa: ndarray | None = None
        self.ba: ndarray | None = None
        self.latent_size = latent_size

        self.dt = .1
        self.temperature = 1.0

    def init_weights(self, features: Features) -> None:
        if self.ws is None:
            Fs = features.state_hand.size
            self.ws = np.random.normal(0.0, 1, size=(Fs, self.latent_size))  # (Fs, L)
            self.bs = np.random.normal(0.0, 1, size=(self.latent_size, 1))  # (L, 1)
            Fa = features.actions.shape[1]
            self.wa = np.random.normal(0.0, 1, size=(Fa, self.latent_size))  # (Fa, L)
            self.ba = np.random.normal(0.0, 1, size=(1, self.latent_size))  # (1, L)
            # (1, L) is better for broadcast with (A, L) action latents

    def update(self, reward: int) -> None:
        assert not self.frozen
        for features, choice_idx, probs in self.trajectory:
            self.update_weights(features, choice_idx, probs, reward)
        self.trajectory = []

    def update_weights(
        self,
        features: Features,
        choice_idx: int,
        probs: ndarray,
        reward: float,
    ) -> None:
        assert self.ws is not None
        assert self.wa is not None
        assert self.bs is not None
        assert self.ba is not None

        dR_dy = softmax_grad(probs, self.temperature, choice_idx, reward)[
            :, None
        ]  # (A, ) -> (A, 1)
        xs = features.state_hand[:, None]  # (Fs, ) -> (Fs, 1)
        xa = features.actions  # (A, Fa)

        # y = la @ ls shapes: (A, ) = (A, L) @ (L, )
        # ls = self.ws @ xs + self.bs  shape would be (Fs, L) @ (Fs, 1) + (L, 1) but i need shape (L, Fs) @ (Fs, 1) + (L, 1) = (L, 1)
        ls = self.ws.T @ xs + self.bs  # (L, 1)
        la = xa @ self.wa + self.ba  # (A, Fa) @ (Fa, L) + (1, L) = (A, L)

        # dR_dy (A, 1)
        # dy_dla = ls (L, 1)
        # dR_dla (A, L) = dR_dy @ dy_dla = dR_dy @ ls shape would be (A, 1) @ (L, 1) but need (A, 1) @ (1, L)
        dR_dla = dR_dy @ ls.T

        # dR_dy (A, 1)
        # dy_dls = la (A, L)
        # dR_dls (L, 1) = dR_dy @ dy_dls = dR_dy @ la shape would be (A, 1) @ (A, L) but need (L, A) @ (A, 1)
        dR_dls = la.T @ dR_dy  # (L, 1)

        # dls_dws = xs
        # dR_dws (Fs, L) = dR_dls @ dls_dws = dR_dls @ xs shape would be (L, 1) @ (Fs, 1) need (Fs, 1) @ (1, L)
        dR_dws = xs @ dR_dls.T

        # dls_dbs = 1
        # dR_dbs (L, 1) = dR_dls @ dls_dws = dR_dls @ 1 shape would be (L, 1) @ 1 need (L, 1)
        dR_dbs = dR_dls

        # dla_dwa = xa
        # dR_dwa (Fa, L) = dR_dla @ dla_dwa = dR_dla @ xa shape would be (A, L) @ (A, Fa) need (Fa, A) @ (A, L)
        dR_dwa = xa.T @ dR_dla

        # dla_dba = 1
        # dR_dbb (1, L) = dR_dla @ dla_dba = dR_dla @ 1 shape would be (A, L) @ 1 need (L, 1) so (A, L).sum(axis = 0)
        dR_dba = dR_dla.sum(axis=0, keepdims=True)

        dt = self.dt
        self.ws += dR_dws * dt
        self.wa += dR_dwa * dt
        self.bs += dR_dbs * dt
        self.ba += dR_dba * dt

    def get_probabilities(self, features: Features) -> ndarray:
        assert self.ws is not None
        assert self.wa is not None
        assert self.bs is not None
        assert self.ba is not None

        xs = features.state_hand[:, None]  # (Fs, ) -> (Fs, 1)
        xa = features.actions  # (A, Fa)

        ls = self.ws.T @ xs + self.bs  # (L, 1)
        la = xa @ self.wa + self.ba  # (A, Fa) @ (Fa, L) = (A, L)
        y = la @ ls
        y = y[:, 0]  # (A, 1) -> (A, )

        return softmax(y, self.temperature)  # (A, L) @ (L, 1) = (A, )

    def save(self, path: str) -> None:
        assert self.ws is not None
        assert self.bs is not None
        assert self.wa is not None
        assert self.ba is not None
        np.savez(
            path,
            version=np.array(1, dtype=int),
            kind=np.array("attention"),
            dt=np.array(self.dt, dtype=float),
            temperature=np.array(self.temperature, dtype=float),
            frozen=np.array(int(self.frozen), dtype=int),
            latent_size=np.array(self.latent_size, dtype=int),
            ws=self.ws,
            bs=self.bs,
            wa=self.wa,
            ba=self.ba,
        )


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
    elif kind == "attention":
        latent_size = int(
            checkpoint["latent_size"]
            if "latent_size" in checkpoint
            else checkpoint["ws"].shape[1]
        )
        agent = AttentionAgent()
        agent.latent_size = latent_size
        agent.ws = checkpoint["ws"]
        agent.bs = checkpoint["bs"]
        agent.wa = checkpoint["wa"]
        agent.ba = checkpoint["ba"]
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
            clone.weights += np.random.normal(
                0.0, perturb_std, size=clone.weights.shape
            )
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
    elif isinstance(agent, AttentionAgent):
        clone = AttentionAgent()
        assert agent.ws is not None
        assert agent.bs is not None
        assert agent.wa is not None
        assert agent.ba is not None
        clone.latent_size = agent.latent_size
        clone.ws = agent.ws.copy()
        clone.bs = agent.bs.copy()
        clone.wa = agent.wa.copy()
        clone.ba = agent.ba.copy()
        if perturb_std > 0:
            clone.ws += np.random.normal(0.0, perturb_std, size=clone.ws.shape)
            clone.bs += np.random.normal(0.0, perturb_std, size=clone.bs.shape)
            clone.wa += np.random.normal(0.0, perturb_std, size=clone.wa.shape)
            clone.ba += np.random.normal(0.0, perturb_std, size=clone.ba.shape)
    else:
        raise ValueError(f"Unsupported agent type for cloning: {type(agent).__name__}")

    clone.dt = agent.dt
    clone.temperature = agent.temperature
    clone.frozen = agent.frozen
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
