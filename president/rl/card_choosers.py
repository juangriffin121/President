from __future__ import annotations

import numpy as np
from numpy import ndarray

from president.nn.layers import Linear, Leaky_Relu, Tanh
from president.nn.network import NeuralNetwork
from president.rl.chooser import CardChooser
from president.rl.features import Features


class LinearChooser(CardChooser):
    def __init__(self, dt: float = 0.6, temperature: float = 3.0) -> None:
        self.weights: ndarray | None = None
        self.dt = dt
        self.temperature = temperature
        self.frozen = False
        self.baseline = 0.0
        self.baseline_lr = 0.05

    def initialize(self, features: Features) -> None:
        if self.weights is None:
            num_features = features.state_hand.size + features.actions.shape[1]
            self.weights = np.random.normal(0.0, 4, size=num_features)

    def update(self, trajectory: list[tuple[Features, int, ndarray]], reward: int) -> float:
        if self.frozen:
            return 0.0
        advantage = reward - self.baseline
        self.baseline += self.baseline_lr * (reward - self.baseline)
        for features, choice_idx, probs in trajectory:
            self._update_weights(features, choice_idx, probs, advantage)
        return float(advantage)

    def _weight_grad(
        self, features: Features, choice_idx: int, probs: ndarray, advantage: float
    ) -> ndarray:
        flat_features = features.as_concatenated()
        grad = _softmax_grad(probs, self.temperature, choice_idx, advantage)
        return grad @ flat_features

    def _update_weights(
        self, features: Features, choice_idx: int, probs: ndarray, reward: float
    ) -> None:
        assert self.weights is not None
        self.weights += self.dt * self._weight_grad(features, choice_idx, probs, reward)

    def update_batch(
        self, games: list[tuple[list[tuple[Features, int, ndarray]], int]]
    ) -> float:
        if self.frozen or not games or self.weights is None:
            return 0.0
        baseline = self.baseline
        grad_accum = np.zeros_like(self.weights)
        for trajectory, reward in games:
            advantage = reward - baseline
            for features, choice_idx, probs in trajectory:
                grad_accum += self._weight_grad(features, choice_idx, probs, advantage)
        self.weights += self.dt * grad_accum / len(games)
        batch_mean_reward = sum(r for _, r in games) / len(games)
        self.baseline += self.baseline_lr * (batch_mean_reward - self.baseline)
        return float(batch_mean_reward - baseline)

    def get_probabilities(self, features: Features) -> ndarray:
        flat_features = features.as_concatenated()
        assert self.weights is not None
        if flat_features.size == 0:
            return np.zeros(flat_features.shape[0], dtype=float)
        scores = flat_features @ self.weights
        return _softmax(scores, self.temperature)

    def save_payload(self) -> dict[str, ndarray]:
        assert self.weights is not None
        payload: dict[str, ndarray] = {
            "version": np.array(1, dtype=int),
            "kind": np.array("linear"),
            "dt": np.array(self.dt, dtype=float),
            "temperature": np.array(self.temperature, dtype=float),
            "frozen": np.array(int(self.frozen), dtype=int),
            "weights": self.weights,
        }
        return payload

    @classmethod
    def load(cls, checkpoint) -> "LinearChooser":
        chooser = cls()
        chooser.weights = checkpoint["weights"]
        chooser.dt = float(checkpoint["dt"])
        chooser.temperature = float(checkpoint["temperature"])
        chooser.frozen = bool(int(checkpoint["frozen"]))
        return chooser

    def clone(self, perturb_std: float = 0.1) -> "LinearChooser":
        clone = LinearChooser()
        assert self.weights is not None
        clone.weights = self.weights.copy()
        clone.dt = self.dt
        clone.temperature = self.temperature
        clone.frozen = self.frozen
        if perturb_std > 0:
            clone.weights += np.random.normal(
                0.0, perturb_std, size=clone.weights.shape
            )
        return clone


class MLPChooser(CardChooser):
    def __init__(self, hidden_layers_sizes: tuple[int, ...], dt:float=0.1, temperature:float=1.0) -> None:
        self.dt = dt
        self.temperature = temperature
        self.frozen = False
        self.baseline = 0.0
        self.baseline_lr = 0.05

        self.hidden_layers_sizes = hidden_layers_sizes
        self.weights: list[ndarray] | None = None
        self.biases: list[ndarray] | None = None
        self.neuron_log: list[tuple[list[ndarray], list[ndarray]]] = []

    def initialize(self, features: Features) -> None:
        if self.weights is None:
            num_features = features.state_hand.size + features.actions.shape[1]
            layer_sizes = (num_features, *self.hidden_layers_sizes, 1)
            self.weights = []
            self.biases = []
            for layer_size, next_layer_size in zip(layer_sizes, layer_sizes[1:]):
                self.weights.append(
                    np.random.normal(0.0, 0.1, size=(layer_size, next_layer_size))
                )
                self.biases.append(np.random.normal(0.0, 0.1, size=(next_layer_size)))

    def update(self, trajectory: list[tuple[Features, int, ndarray]], reward: int) -> float:
        if self.frozen:
            self.neuron_log = []
            return 0.0
        advantage = reward - self.baseline
        self.baseline += self.baseline_lr * (reward - self.baseline)
        for (_, choice_idx, probs), (x_cache, y_cache) in zip(
            trajectory, self.neuron_log
        ):
            self._update_weights(choice_idx, probs, advantage, x_cache, y_cache)
        self.neuron_log = []
        return float(advantage)

    def _weight_grads(
        self,
        choice_idx: int,
        probs: ndarray,
        advantage: float,
        x_cache: list[ndarray],
        y_cache: list[ndarray],
    ) -> tuple[list[ndarray], list[ndarray]]:
        assert self.weights is not None
        dR_dy = _softmax_grad(probs, self.temperature, choice_idx, advantage)[:, None]
        dWs: list[ndarray] = [None] * len(self.weights)  # type: ignore
        dbs: list[ndarray] = [None] * len(self.weights)  # type: ignore
        for i in reversed(range(len(self.weights))):
            w = self.weights[i]
            x = x_cache[i]
            dWs[i] = x.T @ dR_dy
            dbs[i] = dR_dy.sum(axis=0)
            if i > 0:
                dR_dX = dR_dy @ w.T
                y = y_cache[i - 1]
                dR_dy = dR_dX * _leaky_relu_grad(y)
        return dWs, dbs

    def _update_weights(
        self,
        choice_idx: int,
        probs: ndarray,
        reward: float,
        x_cache: list[ndarray],
        y_cache: list[ndarray],
    ) -> None:
        assert self.weights is not None
        assert self.biases is not None
        dWs, dbs = self._weight_grads(choice_idx, probs, reward, x_cache, y_cache)
        for i in range(len(self.weights)):
            self.weights[i] += self.dt * dWs[i]
            self.biases[i] += self.dt * dbs[i]

    def update_batch(
        self, games: list[tuple[list[tuple[Features, int, ndarray]], int]]
    ) -> float:
        if self.frozen or not games or self.weights is None:
            self.neuron_log = []
            return 0.0
        assert self.biases is not None
        baseline = self.baseline
        dW_accum = [np.zeros_like(w) for w in self.weights]
        db_accum = [np.zeros_like(b) for b in self.biases]
        log_iter = iter(self.neuron_log)
        for trajectory, reward in games:
            advantage = reward - baseline
            for _, choice_idx, probs in trajectory:
                x_cache, y_cache = next(log_iter)
                dWs, dbs = self._weight_grads(choice_idx, probs, advantage, x_cache, y_cache)
                for i in range(len(self.weights)):
                    dW_accum[i] += dWs[i]
                    db_accum[i] += dbs[i]
        for i in range(len(self.weights)):
            self.weights[i] += self.dt * dW_accum[i] / len(games)
            self.biases[i] += self.dt * db_accum[i] / len(games)
        batch_mean_reward = sum(r for _, r in games) / len(games)
        self.baseline += self.baseline_lr * (batch_mean_reward - self.baseline)
        self.neuron_log = []
        return float(batch_mean_reward - baseline)

    def get_probabilities(self, features: Features) -> ndarray:
        assert self.weights is not None
        assert self.biases is not None
        flat_features = features.as_concatenated()
        if flat_features.size == 0:
            return np.zeros(flat_features.shape[0], dtype=float)

        x = flat_features
        last_idx = len(self.weights) - 1
        x_cache = []
        y_cache = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x_cache.append(x)
            y = x @ w + b
            y_cache.append(y)
            if i == last_idx:
                x = y
            else:
                x = _leaky_relu(y)

        if not self.frozen:
            self.neuron_log.append((x_cache, y_cache))

        x = x.squeeze(1)
        return _softmax(x, self.temperature)

    def save_payload(self) -> dict[str, ndarray]:
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
        return payload

    @classmethod
    def load(cls, checkpoint) -> "MLPChooser":
        hidden_layers = tuple(int(x) for x in checkpoint["hidden_layers_sizes"].tolist())
        chooser = cls(hidden_layers)
        num_layers = int(checkpoint["num_layers"])
        chooser.weights = [checkpoint[f"w{i}"] for i in range(num_layers)]
        chooser.biases = [checkpoint[f"b{i}"] for i in range(num_layers)]
        chooser.neuron_log = []
        chooser.dt = float(checkpoint["dt"])
        chooser.temperature = float(checkpoint["temperature"])
        chooser.frozen = bool(int(checkpoint["frozen"]))
        return chooser

    def clone(self, perturb_std: float = 0.1) -> "MLPChooser":
        clone = MLPChooser(self.hidden_layers_sizes)
        assert self.weights is not None
        assert self.biases is not None
        clone.weights = [w.copy() for w in self.weights]
        clone.biases = [b.copy() for b in self.biases]
        clone.dt = self.dt
        clone.temperature = self.temperature
        clone.frozen = self.frozen
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


class StateScorerChooser(CardChooser):
    def __init__(self, dt:float=0.2, temperature:float = 3) -> None:
        self.w: ndarray | None = None
        self.b: ndarray | None = None
        self.dt = dt
        self.temperature = temperature
        self.frozen = False
        self.baseline = 0.0
        self.baseline_lr = 0.05

    def initialize(self, features: Features) -> None:
        if self.w is None:
            Fs = features.state_hand.size
            Fa = features.actions.shape[1]
            self.w = np.random.normal(0.0, 2, size=(Fa, Fs))
            self.b = np.random.normal(0.0, 2, size=(Fa, 1))

    def update(self, trajectory: list[tuple[Features, int, ndarray]], reward: int) -> float:
        if self.frozen:
            return 0.0
        advantage = reward - self.baseline
        self.baseline += self.baseline_lr * (reward - self.baseline)
        for features, choice_idx, probs in trajectory:
            self._update_weights(features, choice_idx, probs, advantage)
        return float(advantage)

    def _weight_grad(
        self, features: Features, choice_idx: int, probs: ndarray, advantage: float
    ) -> tuple[ndarray, ndarray]:
        dR_dy = _softmax_grad(probs, self.temperature, choice_idx, advantage)[:, None]
        xs = features.state_hand[:, None]
        xa = features.actions
        dR_dz = xa.T @ dR_dy
        dR_db = dR_dz
        dR_dw = dR_dz @ xs.T
        return dR_dw, dR_db

    def _update_weights(
        self, features: Features, choice_idx: int, probs: ndarray, reward: float
    ) -> None:
        assert self.w is not None
        assert self.b is not None
        dR_dw, dR_db = self._weight_grad(features, choice_idx, probs, reward)
        self.w += dR_dw * self.dt
        self.b += dR_db * self.dt

    def update_batch(
        self, games: list[tuple[list[tuple[Features, int, ndarray]], int]]
    ) -> float:
        if self.frozen or not games or self.w is None:
            return 0.0
        assert self.b is not None
        baseline = self.baseline
        dw_accum = np.zeros_like(self.w)
        db_accum = np.zeros_like(self.b)
        for trajectory, reward in games:
            advantage = reward - baseline
            for features, choice_idx, probs in trajectory:
                dw, db = self._weight_grad(features, choice_idx, probs, advantage)
                dw_accum += dw
                db_accum += db
        self.w += self.dt * dw_accum / len(games)
        self.b += self.dt * db_accum / len(games)
        batch_mean_reward = sum(r for _, r in games) / len(games)
        self.baseline += self.baseline_lr * (batch_mean_reward - self.baseline)
        return float(batch_mean_reward - baseline)

    def get_probabilities(self, features: Features) -> ndarray:
        assert self.w is not None
        assert self.b is not None

        xs = features.state_hand[:, None]
        xa = features.actions

        z = self.w @ xs + self.b
        y = xa @ z
        y = y[:, 0]

        return _softmax(y, self.temperature)

    def save_payload(self) -> dict[str, ndarray]:
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
        return payload

    @classmethod
    def load(cls, checkpoint) -> "StateScorerChooser":
        chooser = cls()
        chooser.w = checkpoint["w"]
        chooser.b = checkpoint["b"]
        chooser.dt = float(checkpoint["dt"])
        chooser.temperature = float(checkpoint["temperature"])
        chooser.frozen = bool(int(checkpoint["frozen"]))
        return chooser

    def clone(self, perturb_std: float = 0.1) -> "StateScorerChooser":
        clone = StateScorerChooser()
        assert self.w is not None
        assert self.b is not None
        clone.w = self.w.copy()
        clone.b = self.b.copy()
        clone.dt = self.dt
        clone.temperature = self.temperature
        clone.frozen = self.frozen
        if perturb_std > 0:
            clone.w += np.random.normal(0.0, perturb_std, size=clone.w.shape)
            clone.b += np.random.normal(0.0, perturb_std, size=clone.b.shape)
        return clone


class ActorCriticChooser(CardChooser):
    def __init__(self, latent_dim: int, critic_weight: float = 0.1, dt: float=0.1, temperature: float = 5.0):
        self.latent_dim = latent_dim
        self.critic_weight: float = critic_weight
        self.state_encoder_cache: list[list[ndarray]] = []
        self.action_encoder_cache: list[list[ndarray]] = []
        self.latent_cache: list[tuple[ndarray, ndarray]] = []
        self.critic_cache: list[list[ndarray]] = []
        self.state_values: list[ndarray] = []

        self.dt = dt
        self.temperature = temperature
        self.frozen = False
        self.baseline = 0.0
        self.baseline_lr = 0.05

    def _make_encoder(self, input_size):
        return NeuralNetwork(
            input_size,
            [
                Linear(self.latent_dim),
                Leaky_Relu(),
            ],
        )

    def initialize(self, features: Features) -> None:
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
        xs = features.state_hand[:, None]
        xa = features.actions.T

        ls, state_encoder_cache = self.state_encoder.forward(xs)
        la, action_encoder_cache = self.action_encoder.forward(xa)
        v, critic_cache = self.critic.forward(ls)
        v = 2 * v

        if not self.frozen:
            self.state_encoder_cache.append(state_encoder_cache)
            self.action_encoder_cache.append(action_encoder_cache)
            self.latent_cache.append((ls, la))
            self.critic_cache.append(critic_cache)
            self.state_values.append(v)

        logits = la.T @ ls
        probs = _softmax(logits[:, 0], self.temperature)
        return probs

    def update(self, trajectory: list[tuple[Features, int, ndarray]], reward: int) -> float:
        if self.frozen:
            self._clear_cache()
            return 0.0
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
            trajectory,
            self.state_encoder_cache,
            self.action_encoder_cache,
            self.latent_cache,
            self.critic_cache,
            self.state_values,
        ):
            dt = self.dt
            advantage = reward - float(value_estimation[0][0])
            dR_dy = _softmax_grad(probs, self.temperature, choice_idx, advantage)[
                :, None
            ]
            dL_dv = 4 * (value_estimation - reward) # two 2s, one from the 2*tanh and one from the loss (v-r)^2
            dL_dls = self.critic.backward(dL_dv, dt, critic_cache)

            dR_dls = la @ dR_dy
            dR_dla = ls @ dR_dy.T

            self.state_encoder.backward(
                -dR_dls + self.critic_weight * dL_dls, dt, state_encoder_cache
            )
            self.action_encoder.backward(-dR_dla, dt, action_encoder_cache)

        self._clear_cache()
        return float(advantage_wc)

    def update_batch(
        self, games: list[tuple[list[tuple[Features, int, ndarray]], int]]
    ) -> float:
        if self.frozen or not games:
            self._clear_cache()
            return 0.0
        baseline = self.baseline
        dt = self.dt / len(games)

        critic_grads = None
        state_grads = None
        action_grads = None

        cache_iter = zip(
            self.state_encoder_cache,
            self.action_encoder_cache,
            self.latent_cache,
            self.critic_cache,
            self.state_values,
        )

        for trajectory, reward in games:
            for (_, choice_idx, probs), cache_entry in zip(trajectory, cache_iter):
                (
                    state_encoder_cache,
                    action_encoder_cache,
                    (ls, la),
                    critic_cache,
                    value_estimation,
                ) = cache_entry

                step_advantage = reward - float(value_estimation[0][0])
                dR_dy = _softmax_grad(
                    probs, self.temperature, choice_idx, step_advantage
                )[:, None]
                dL_dv = 4 * (value_estimation - reward)
                dL_dls, c_grads = self.critic.backward(
                    dL_dv, dt, critic_cache, accumulate=True
                )

                dR_dls = la @ dR_dy
                dR_dla = ls @ dR_dy.T

                _, s_grads = self.state_encoder.backward(
                    -dR_dls + self.critic_weight * dL_dls,
                    dt,
                    state_encoder_cache,
                    accumulate=True,
                )
                _, a_grads = self.action_encoder.backward(
                    -dR_dla, dt, action_encoder_cache, accumulate=True
                )

                critic_grads = _accum_grads(critic_grads, c_grads)
                state_grads = _accum_grads(state_grads, s_grads)
                action_grads = _accum_grads(action_grads, a_grads)

        if critic_grads is not None:
            self.critic.apply_grads(critic_grads, dt)
            self.state_encoder.apply_grads(state_grads, dt)
            self.action_encoder.apply_grads(action_grads, dt)

        batch_mean_reward = sum(r for _, r in games) / len(games)
        self.baseline += self.baseline_lr * (batch_mean_reward - self.baseline)
        self._clear_cache()
        return float(batch_mean_reward - baseline)

    def _clear_cache(self) -> None:
        self.state_encoder_cache = []
        self.action_encoder_cache = []
        self.latent_cache = []
        self.critic_cache = []
        self.state_values = []

    def save_payload(self) -> dict[str, ndarray]:
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
        return payload

    @classmethod
    def load(cls, checkpoint) -> "ActorCriticChooser":
        latent_dim = int(checkpoint["latent_dim"])
        critic_weight = (
            float(checkpoint["critic_weight"])
            if "critic_weight" in checkpoint
            else 0.5
        )
        chooser = cls(latent_dim, critic_weight=critic_weight)
        Fs = int(checkpoint["state_input_size"])
        Fa = int(checkpoint["action_input_size"])

        chooser.state_encoder = chooser._make_encoder(Fs)
        chooser.action_encoder = chooser._make_encoder(Fa)
        chooser.critic = NeuralNetwork(chooser.latent_dim, [Linear(1), Tanh()])
        chooser.state_encoder.initialize()
        chooser.action_encoder.initialize()
        chooser.critic.initialize()

        def extract_payload(prefix: str) -> dict[str, ndarray]:
            num = int(checkpoint[f"{prefix}__num_linear_layers"])
            payload = {"num_linear_layers": checkpoint[f"{prefix}__num_linear_layers"]}
            for i in range(num):
                payload[f"w{i}"] = checkpoint[f"{prefix}__w{i}"]
                payload[f"b{i}"] = checkpoint[f"{prefix}__b{i}"]
            return payload

        chooser.state_encoder.load(extract_payload("state_enc"))
        chooser.action_encoder.load(extract_payload("action_enc"))
        chooser.critic.load(extract_payload("critic"))
        chooser.dt = float(checkpoint["dt"])
        chooser.temperature = float(checkpoint["temperature"])
        chooser.frozen = bool(int(checkpoint["frozen"]))
        return chooser

    def clone(self, perturb_std: float = 0.0) -> "ActorCriticChooser":
        clone = ActorCriticChooser(self.latent_dim, critic_weight=self.critic_weight)
        assert hasattr(self, "state_encoder")
        assert hasattr(self, "action_encoder")
        assert hasattr(self, "critic")
        clone.state_encoder = self.state_encoder.clone(perturb_std=perturb_std)
        clone.action_encoder = self.action_encoder.clone(perturb_std=perturb_std)
        clone.critic = self.critic.clone(perturb_std=perturb_std)
        clone.dt = self.dt
        clone.temperature = self.temperature
        clone.frozen = self.frozen
        return clone


def _leaky_relu(x: ndarray) -> ndarray:
    return np.where(x > 0, x, 0.1 * x)


def _leaky_relu_grad(y: ndarray) -> ndarray:
    return np.where(y > 0, 1, 0.1)


def _softmax(x: ndarray, temp: float) -> ndarray:
    x = x / max(temp, 1e-6)
    max_x = np.max(x) if x.size else 0
    exp_x = np.exp(x - max_x)
    total = exp_x.sum()
    if total <= 0:
        return np.zeros_like(exp_x)
    return exp_x / total


def _softmax_grad(y: ndarray, temp: float, choice_idx: int, reward: float) -> ndarray:
    one_hot = np.zeros(y.shape[0])
    one_hot[choice_idx] = 1
    return reward * (one_hot - y) / max(temp, 1e-6)

def _accum_grads(acc, new):
    if acc is None:
        return new
    return [(a0 + b0, a1 + b1) for (a0, a1), (b0, b1) in zip(acc, new)]
