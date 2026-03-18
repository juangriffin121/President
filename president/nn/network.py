import numpy as np

from president.nn.layers import Layer, Linear


class NeuralNetwork:
    def __init__(self, input_size: int, layers: list[Layer]):
        self.layers = layers
        self.input_size = input_size

    def initialize(self):
        input_size = self.input_size
        for layer in self.layers:
            layer.set_input_size(input_size)
            layer.initialize()
            input_size = layer.output_size
        self.output_size = input_size

    def forward(self, input_data: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        output = input_data
        cache = []
        for layer in self.layers:
            output, layer_cache = layer.forward(output)
            cache.append(layer_cache)
        return output, cache

    def backward(
        self, grad_output: np.ndarray, dt, cache: list[np.ndarray]
    ) -> np.ndarray:
        for layer, layer_cache in zip(reversed(self.layers), reversed(cache)):
            grad_output = layer.backward(grad_output, dt, layer_cache)
        return grad_output

    def __getitem__(self, index):
        return self.layers[index]

    def __len__(self):
        return len(self.layers)

    def __str__(self):
        txt = ""
        for layer in self.layers:
            txt += f"{str(layer)}\n"
        return txt

    def set_input_size(self, input_size):
        if hasattr(self, "input_size"):
            if input_size != self.input_size:
                print("Changing the input size")
        self.input_size = input_size

    def __call__(self, *prev_layers):
        self.prev_layers = prev_layers
        return self

    def freeze(self):
        for layer in self.layers:
            layer.freeze()

    def unfreeze(self):
        for layer in self.layers:
            layer.unfreeze()

    def clone(self, perturb_std: float = 0.0):
        clone_layers = [layer.clone(perturb_std) for layer in self.layers]
        clone = NeuralNetwork(self.input_size, clone_layers)
        clone.output_size = self.output_size
        return clone

    def save_payload(self) -> dict[str, np.ndarray]:
        weights: list[np.ndarray] = []
        biases: list[np.ndarray] = []
        for layer in self.layers:
            if hasattr(layer, "pesos"):
                assert isinstance(layer, Linear)
                weights.append(layer.pesos)
                biases.append(layer.sesgos)
        payload: dict[str, np.ndarray] = {
            "num_linear_layers": np.array(len(weights), dtype=int),
        }
        for idx, w in enumerate(weights):
            payload[f"w{idx}"] = w
        for idx, b in enumerate(biases):
            payload[f"b{idx}"] = b
        return payload

    def load(self, payload: dict[str, np.ndarray]) -> None:
        num_layers = int(payload["num_linear_layers"])
        assert len(self.layers) == num_layers
        weights = [payload[f"w{i}"] for i in range(num_layers)]
        biases = [payload[f"b{i}"] for i in range(num_layers)]
        w_idx = 0
        for layer in self.layers:
            if hasattr(layer, "pesos"):
                if w_idx >= num_layers:
                    raise ValueError("Not enough weights/biases for network layers")
                assert isinstance(layer, Linear)
                layer.pesos = weights[w_idx]
                layer.sesgos = biases[w_idx]
                w_idx += 1
        if w_idx != num_layers:
            raise ValueError("Unused weights/biases provided to network load")
