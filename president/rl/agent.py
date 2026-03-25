from __future__ import annotations
from abc import ABC, abstractmethod
from numpy import ndarray, number
import numpy as np
from president.card import Card, Joker
from president.nn.layers import Layer, Linear, Leaky_Relu, Tanh
from president.nn.network import NeuralNetwork
from president.rl.features import NUM_CARD_FEATS, Features, get_card_features, get_features, get_hand_features
from president.state import GlobalState, PlayerState
from president.rules import valid_choice
from president.utils import possible_sets
from president.ranking import get_num, order_num


class Agent(ABC):
    def __init__(self) -> None:
        self.trajectory: list[tuple[Features, int, ndarray]] = []
        self.worst_chooser_cache: tuple[list[int], list[ndarray], list[ndarray]] = (
            [],
            [],
            [],
        )
        self.frozen = False
        self.dt = 0.5  # Redefined in subclasses
        self.temperature = 5.0  # Redefined in subclasses
        self.baseline = 0.0
        self.baseline_lr = 0.05

    def choose_cards(
        self, state: GlobalState, player_state
    ) -> list[Card | Joker] | None:
        last_played = state.played[-1] if state.played else None
        valid_actions = self.get_valid_actions(last_played, player_state.hand)
        features = get_features(state, player_state, valid_actions)
        self.initialize(features)
        probs = self.get_probabilities(features)
        choice_idx, choice = self.choose(valid_actions, probs)
        if not self.frozen:
            self.trajectory.append((features, choice_idx, probs))
        return choice

    def initialize_worst_chooser(self):
        if hasattr(self, "worst_chooser"):
            return
        self.worst_chooser = NeuralNetwork(NUM_CARD_FEATS, [Linear(1)])
        self.worst_chooser.initialize()

    @abstractmethod
    def initialize(self, features: Features) -> None:
        self.initialize_worst_chooser()

    def choose_worst(self, count, hand) -> list[Card | Joker]:
        card_feats = get_card_features(hand) # (C, Fc)
        scores, cache = self.worst_chooser.forward(card_feats.T) # (Fc, C) -> (1, C)
        probs = softmax(scores.T[:, 0], self.temperature)
        idx1 = np.random.choice(len(probs), p=probs)
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

        if not self.frozen:
            worst_chosen = [idx1]
            if idx2 is not None:
                worst_chosen.append((idx2)) 

            self.worst_chooser_cache = (worst_chosen, probabilities, cache)

        return cards

    def _update_worst_chooser(self, advantage: float) -> None:
        if not hasattr(self, "worst_chooser"):
            self.worst_chooser_cache = ([], [], [])
            return
        worst_chosen, probabilities, cache = self.worst_chooser_cache
        if not worst_chosen:
            return
        assert not self.frozen
        for choice_idx, probs in zip(worst_chosen, probabilities):
            grad = softmax_grad(probs, self.temperature, choice_idx, advantage)
            grad_output = grad[None, :]  # (1, C)
            self.worst_chooser.backward(grad_output, self.dt, cache)
        self.worst_chooser_cache = ([], [], [])

    def _add_worst_payload(self, payload: dict[str, ndarray]) -> None:
        if not hasattr(self, "worst_chooser"):
            return
        for key, value in self.worst_chooser.save_payload().items():
            payload[f"worst__{key}"] = value

    def _load_worst_payload(self, checkpoint) -> None:
        key = "worst__num_linear_layers"
        if key not in checkpoint:
            return
        self.initialize_worst_chooser()
        num = int(checkpoint[key])
        payload = {"num_linear_layers": checkpoint[key]}
        for i in range(num):
            payload[f"w{i}"] = checkpoint[f"worst__w{i}"]
            payload[f"b{i}"] = checkpoint[f"worst__b{i}"]
        self.worst_chooser.load(payload)

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

    @abstractmethod
    def clone(self,perturb_std: float = 0.1) -> Agent:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, checkpoint) -> Agent:
        raise NotImplementedError


class LinearAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.weights: ndarray | None = None
        self.dt = 0.6
        self.temperature = 3.0

    def initialize(self, features: Features) -> None:
        super().initialize(features)
        if self.weights is None:
            num_features = features.state_hand.size + features.actions.shape[1]
            self.weights = np.random.normal(0.0, 4, size=num_features)

    def update(self, reward: int) -> None:
        advantage = reward - self.baseline
        self.baseline += self.baseline_lr * (reward - self.baseline)
        assert not self.frozen
        for features, choice_idx, probs in self.trajectory:
            self.update_weights(features, choice_idx, probs, advantage)
        self._update_worst_chooser(advantage)
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
        payload: dict[str, ndarray] = {
            "version": np.array(1, dtype=int),
            "kind": np.array("linear"),
            "dt": np.array(self.dt, dtype=float),
            "temperature": np.array(self.temperature, dtype=float),
            "frozen": np.array(int(self.frozen), dtype=int),
            "weights": self.weights,
        }
        self._add_worst_payload(payload)
        np.savez(path, **payload)

    @classmethod
    def load(cls, checkpoint) -> Agent:
        agent = cls()
        agent.weights = checkpoint["weights"]
        agent._load_worst_payload(checkpoint)
        return agent

    def clone(self, perturb_std: float = 0.1) -> Agent:
        clone = LinearAgent()
        assert self.weights is not None
        clone.weights = self.weights.copy()
        if perturb_std > 0:
            clone.weights += np.random.normal(
                0.0, perturb_std, size=clone.weights.shape
            )
        return clone
    
        


class MLPAgent(Agent):
    def __init__(self, hidden_layers_sizes: tuple[int, ...]) -> None:
        super().__init__()

        self.dt = 0.1
        self.temperature = 1.0

        self.hidden_layers_sizes = hidden_layers_sizes
        self.weights: list[ndarray] | None = None
        self.biases: list[ndarray] | None = None
        self.neuron_log: list[tuple[list[ndarray], list[ndarray]]] = []

    def initialize(self, features: Features) -> None:
        super().initialize(features)
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
        advantage = reward - self.baseline
        self.baseline += self.baseline_lr * (reward - self.baseline)
        for (_, choice_idx, probs), (x_cache, y_cache) in zip(
            self.trajectory, self.neuron_log
        ):
            self.update_weights(choice_idx, probs, advantage, x_cache, y_cache)
        self._update_worst_chooser(advantage)
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
        self._add_worst_payload(payload)
        np.savez(path, **payload)

    @classmethod
    def load(cls, checkpoint) -> Agent:
        hidden_layers = tuple(
            int(x) for x in checkpoint["hidden_layers_sizes"].tolist()
        )
        agent = cls(hidden_layers)
        num_layers = int(checkpoint["num_layers"])
        agent.weights = [checkpoint[f"w{i}"] for i in range(num_layers)]
        agent.biases = [checkpoint[f"b{i}"] for i in range(num_layers)]
        agent.neuron_log = []
        agent._load_worst_payload(checkpoint)
        return agent

    def clone(self, perturb_std: float = 0.1) -> Agent:
        clone = MLPAgent(self.hidden_layers_sizes)
        assert self.weights is not None
        assert self.biases is not None
        clone.weights = [w.copy() for w in self.weights]
        clone.biases = [b.copy() for b in self.biases]
        if perturb_std > 0:
            clone.weights = [
                w + np.random.normal(0.0, perturb_std, size=w.shape)
                for w in clone.weights
            ]
            clone.biases = [
                b + np.random.normal(0.0, perturb_std, size=b.shape)
                for b in clone.biases
            ]
        return clone


class StateScorerAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.w: ndarray | None = None
        self.b: ndarray | None = None

        self.dt = 0.5
        self.temperature = 1.0

    def initialize(self, features: Features) -> None:
        super().initialize(features)
        if self.w is None:
            Fs = features.state_hand.size
            Fa = features.actions.shape[1]
            self.w = np.random.normal(0.0, 2, size=(Fa, Fs))  # (Fa, Fs)
            self.b = np.random.normal(0.0, 2, size=(Fa, 1))  # (Fa, 1)

    def update(self, reward: int) -> None:
        assert not self.frozen
        advantage = reward - self.baseline
        self.baseline += self.baseline_lr * (reward - self.baseline)
        for features, choice_idx, probs in self.trajectory:
            self.update_weights(features, choice_idx, probs, advantage)
        self._update_worst_chooser(advantage)
        self.trajectory = []

    def update_weights(
        self,
        features: Features,
        choice_idx: int,
        probs: ndarray,
        reward: float,
    ) -> None:
        assert self.w is not None
        assert self.b is not None

        dR_dy = softmax_grad(probs, self.temperature, choice_idx, reward)[
            :, None
        ]  # (A, ) -> (A, 1)
        xs = features.state_hand[:, None]  # (Fs, ) -> (Fs, 1)
        xa = features.actions  # (A, Fa)

        # If not linear
        # z = leaky_relu(self.w @ xs + self.b)  # (Fa, Fs) @ (Fs, 1) = (Fa, 1)

        dR_dz = xa.T @ dR_dy  # (Fa, A) @ (A, 1) = (Fa, 1)
        # dR_dz = dR_dz * leaky_relu_grad(z) # (Fa, 1)
        dR_db = dR_dz  # (Fa, 1)
        # dR_dw = dR_dz dz_dw
        dR_dw = dR_dz @ xs.T  #  (Fa, 1) @ (1, Fs)

        dt = self.dt
        self.w += dR_dw * dt
        self.b += dR_db * dt

    def get_probabilities(self, features: Features) -> ndarray:
        assert self.w is not None
        assert self.b is not None

        xs = features.state_hand[:, None]  # (Fs, ) -> (Fs, 1)
        xa = features.actions  # (A, Fa)

        # If not linear
        # z = leaky_relu(self.w @ xs + self.b)  # (Fa, Fs) @ (Fs, 1) = (Fa, 1)
        z = self.w @ xs + self.b  # (Fa, Fs) @ (Fs, 1) = (Fa, 1)
        y = xa @ z  # (A, Fa) @ (Fa, 1)
        y = y[:, 0]  # (A, 1) -> (A, )

        return softmax(y, self.temperature)

    def save(self, path: str) -> None:
        assert self.w is not None
        assert self.b is not None
        payload: dict[str, ndarray] = {
            "version": np.array(1, dtype=int),
            "kind": np.array("state_scorer"),
            "dt": np.array(self.dt, dtype=float),
            "temperature": np.array(self.temperature, dtype=float),
            "frozen": np.array(int(self.frozen), dtype=int),
            "w": self.w,
            "b": self.b,
        }
        self._add_worst_payload(payload)
        np.savez(path, **payload)

    @classmethod
    def load(cls, checkpoint) -> Agent:
        agent = cls()
        agent.w = checkpoint["w"]
        agent.b = checkpoint["b"]
        agent._load_worst_payload(checkpoint)
        return agent

    def clone(self, perturb_std: float = 0.1) -> Agent:
        clone = StateScorerAgent()
        assert self.w is not None
        assert self.b is not None
        clone.w = self.w.copy()
        clone.b = self.b.copy()
        if perturb_std > 0:
            clone.w += np.random.normal(0.0, perturb_std, size=clone.w.shape)
            clone.b += np.random.normal(0.0, perturb_std, size=clone.b.shape)
        return clone


class ActorCritic(Agent):
    def __init__(self, latent_dim: int, critic_weight: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.critic_weight: float = critic_weight
        self.state_encoder_cache: list[list[ndarray]] = []
        self.action_encoder_cache: list[list[ndarray]] = []
        self.latent_cache: list[tuple[ndarray, ndarray]] = []
        self.critic_cache: list[list[ndarray]] = []
        self.state_values: list[ndarray] = (
            []
        )  # ndarray to make it easy to plug into backwards but single float

    def _make_encoder(self, input_size):
        return NeuralNetwork(
            input_size,
            [
                # Linear(2*input_size),
                # Leaky_Relu(),
                Linear(self.latent_dim),
                Leaky_Relu(),
            ],
        )

    def initialize(self, features: Features) -> None:
        super().initialize(features)
        if hasattr(self, "state_encoder"):
            return
        Fs = features.state_hand.size
        Fa = features.actions.shape[1]
        self.state_encoder = self._make_encoder(Fs) 
        self.action_encoder = self._make_encoder(Fa) 
        self.critic = NeuralNetwork(self.latent_dim, [Linear(1), Tanh()])

        self.state_encoder.initialize()
        self.action_encoder.initialize()
        self.critic.initialize()

    def get_probabilities(self, features: Features) -> ndarray:
        xs = features.state_hand[:, None]  # (Fs, 1)
        xa = features.actions.T  # (Fa, A)

        ls, state_encoder_cache = self.state_encoder.forward(xs)  # (L, 1)
        la, action_encoder_cache = self.action_encoder.forward(xa)  # (L, A)
        v, critic_cache = self.critic.forward(ls)
        v = 2 * v

        if not self.frozen:
            self.state_encoder_cache.append(state_encoder_cache)
            self.action_encoder_cache.append(action_encoder_cache)
            self.latent_cache.append((ls, la))
            self.critic_cache.append(critic_cache)
            self.state_values.append(v)

        logits = la.T @ ls  # (A, L) @ (L, 1) = (A, 1)
        probs = softmax(logits[:, 0], self.temperature)
        return probs

    def update(self, reward: int) -> None:
        advantage_wc = reward - self.baseline
        self.baseline += self.baseline_lr * (reward - self.baseline)
        for (
            (_, choice_idx, probs),
            state_encoder_cache,
            action_encoder_cache,
            (ls, la),
            critic_cache,
            value_estimation,
        ) in zip(
            self.trajectory,
            self.state_encoder_cache,
            self.action_encoder_cache,
            self.latent_cache,
            self.critic_cache,
            self.state_values,
        ):

            # L = critic_loss
            # R = actor reward

            dt = self.dt
            advantage = reward - float(value_estimation[0][0])
            dR_dy = softmax_grad(probs, self.temperature, choice_idx, advantage)[
                :, None
            ]  # (A, 1)
            dL_dv = 2 * (
                value_estimation - reward
            )  # MSE for critic loss (/2 because Tanh returns [-2, 2] )
            dL_dls = self.critic.backward(dL_dv, dt, critic_cache)

            dR_dls = la @ dR_dy  # (L, A) @ (A, 1) = (L, 1)
            dR_dla = ls @ dR_dy.T   # (L, 1) @ (1, A) = (L, A)

            # print(f"{la.shape} @ {dR_dy.shape} = {dR_dls.shape}")
            # print(f"{ls.shape} @ {dR_dy.T.shape} = {dR_dla.shape}")


            self.state_encoder.backward(
                -dR_dls + self.critic_weight * dL_dls, dt, state_encoder_cache
            )  # negative because backward does gradient descent
            self.action_encoder.backward(-dR_dla, dt, action_encoder_cache)

        self._update_worst_chooser(advantage_wc)
        self.clear_cache()

    def clear_cache(self):
        self.trajectory = []
        self.state_encoder_cache = []
        self.action_encoder_cache = []
        self.latent_cache = []
        self.critic_cache = []
        self.state_values = []

    def save(self, path: str) -> None:
        assert hasattr(self, "state_encoder")
        assert hasattr(self, "action_encoder")
        assert hasattr(self, "critic")
        payload: dict[str, ndarray] = {
            "version": np.array(1, dtype=int),
            "kind": np.array("actor_critic"),
            "dt": np.array(self.dt, dtype=float),
            "temperature": np.array(self.temperature, dtype=float),
            "frozen": np.array(int(self.frozen), dtype=int),
            "latent_dim": np.array(self.latent_dim, dtype=int),
            "critic_weight": np.array(self.critic_weight, dtype=float),
            "state_input_size": np.array(self.state_encoder.input_size, dtype=int),
            "action_input_size": np.array(self.action_encoder.input_size, dtype=int),
        }

        for key, nn_payload in self.state_encoder.save_payload().items():
            payload[f"state_enc__{key}"] = nn_payload
        for key, nn_payload in self.action_encoder.save_payload().items():
            payload[f"action_enc__{key}"] = nn_payload
        for key, nn_payload in self.critic.save_payload().items():
            payload[f"critic__{key}"] = nn_payload
        self._add_worst_payload(payload)
        np.savez(path, **payload)

    @classmethod
    def load(cls, checkpoint) -> Agent:
        latent_dim = int(checkpoint["latent_dim"])
        critic_weight = (
            float(checkpoint["critic_weight"])
            if "critic_weight" in checkpoint
            else 0.5
        )
        agent = cls(latent_dim, critic_weight=critic_weight)
        Fs = int(checkpoint["state_input_size"])
        Fa = int(checkpoint["action_input_size"])

        agent.state_encoder = agent._make_encoder(Fs)
        agent.state_encoder = agent._make_encoder(Fa)
        agent.critic = NeuralNetwork(agent.latent_dim, [Linear(1), Tanh()])
        agent.state_encoder.initialize()
        agent.action_encoder.initialize()
        agent.critic.initialize()

        def extract_payload(prefix: str) -> dict[str, ndarray]:
            num = int(checkpoint[f"{prefix}__num_linear_layers"])
            payload = {"num_linear_layers": checkpoint[f"{prefix}__num_linear_layers"]}
            for i in range(num):
                payload[f"w{i}"] = checkpoint[f"{prefix}__w{i}"]
                payload[f"b{i}"] = checkpoint[f"{prefix}__b{i}"]
            return payload

        agent.state_encoder.load(extract_payload("state_enc"))
        agent.action_encoder.load(extract_payload("action_enc"))
        agent.critic.load(extract_payload("critic"))
        agent._load_worst_payload(checkpoint)
        return agent

    def clone(self, perturb_std: float = 0.0) -> Agent:
        clone = ActorCritic(self.latent_dim, critic_weight=self.critic_weight)
        assert hasattr(self, "state_encoder")
        assert hasattr(self, "action_encoder")
        assert hasattr(self, "critic")
        clone.state_encoder = self.state_encoder.clone(perturb_std=perturb_std)
        clone.action_encoder = self.action_encoder.clone(perturb_std=perturb_std)
        clone.critic = self.critic.clone(perturb_std=perturb_std)
        return clone


def load_agent(path: str) -> Agent:
    checkpoint = np.load(path, allow_pickle=False)
    kind = str(checkpoint["kind"])
    match kind:
        case "linear":
            agent = LinearAgent.load(checkpoint)
        case "mlp":
            agent = MLPAgent.load(checkpoint)
        case "state_scorer":
            agent = StateScorerAgent.load(checkpoint)
        case "actor_critic":
            agent = ActorCritic.load(checkpoint)
        case _:
            raise ValueError(f"Unknown agent kind in checkpoint: {kind}")

    agent.dt = float(checkpoint["dt"])
    agent.temperature = float(checkpoint["temperature"])
    agent.frozen = bool(int(checkpoint["frozen"]))

    return agent

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


def tanh_grad(y: ndarray) -> ndarray:
    return 1 - y**2
