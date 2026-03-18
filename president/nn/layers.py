import numpy as np
from numpy import ndarray


class Layer:
    def __init__(self):
        self.frozen = False
        self.output_size: int

    def forward(self, Input):
        raise NotImplementedError("forward method must be implemented in each layer")

    def backward(self, grad_output, dt, cache):
        raise NotImplementedError("backward method must be implemented in each layer")

    def clone(self, perturb_std: float):
        raise NotImplementedError("clone method must be implemented in each layer")

    def initialize(self):
        raise NotImplementedError("initialize method must be implemented in each layer")

    def set_input_size(self, input_size):
        self.input_size = input_size

    def __str__(self):
        return f"{self.__class__.__name__}\n\t{self.input_size}->{self.output_size}"

    def __call__(self, *prev_layers):
        self.prev_layers = prev_layers
        return self

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False


class Linear(Layer):
    def __init__(self, output_size, weight_var=0.1, bias_var=0.1):
        super().__init__()
        self.output_size = output_size
        self.weight_var = weight_var
        self.bias_var = bias_var

    def initialize(self):
        if hasattr(self, "input_size"):
            self.pesos = self.weight_var * np.random.randn(
                self.output_size, self.input_size
            )  # (O, I)
            self.sesgos = self.bias_var * np.random.randn(self.output_size, 1)  # (O, 1)
        else:
            raise ValueError(
                "Input size not set. Call set_input_size before initializing."
            )

    def forward(self, Input: ndarray) -> tuple[ndarray, ndarray]:
        return (
            self.pesos @ Input + self.sesgos,
            Input,
        )  # (O, I) @ (I, :) + (O, 1) = (O, :)

    def backward(self, grad_output: ndarray, dt, cache: ndarray):
        input_ = cache  # (I, :)
        if not hasattr(self, "frozen") or not self.frozen:
            grad_pesos = grad_output @ input_.T  # (O, :) @ (:, I)
            grad_sesgos = grad_output.sum(axis=1, keepdims=True)  # (O, :) -> (O, 1)
            self.pesos -= grad_pesos * dt
            self.sesgos -= grad_sesgos * dt
        grad_input = self.pesos.T @ grad_output  # (I, O) @ (O, :) = (I, :)
        return grad_input

    def clone(self, perturb_std: float = 0.0):
        clone = Linear(self.output_size, self.weight_var, self.bias_var)
        clone.input_size = self.input_size
        clone.output_size = self.output_size
        if hasattr(self, "pesos"):
            clone.pesos = self.pesos + np.random.normal(
                0.0, perturb_std, size=self.pesos.shape
            )

        if hasattr(self, "sesgos"):
            clone.sesgos = self.sesgos + np.random.normal(
                0.0, perturb_std, size=self.sesgos.shape
            )
        clone.frozen = self.frozen
        return clone


class Activation(Layer):
    def __init__(self, act_fun, dev_act_fun):
        super().__init__()
        self.act_fun = act_fun
        self.dev_act_fun = dev_act_fun

    def initialize(self):
        if hasattr(self, "input_size"):
            self.output_size = self.input_size
        else:
            raise ValueError(
                "Input size not set. Call set_input_size before initializing."
            )

    def forward(self, Input: ndarray) -> tuple[ndarray, ndarray]:
        return self.act_fun(Input), Input

    def backward(self, grad_output, dt, cache):
        input_ = cache
        return grad_output * self.dev_act_fun(input_)


def leaky_relu(x: ndarray) -> ndarray:
    return np.where(x > 0, x, 0.1 * x)


def dev_leaky_relu(y: ndarray) -> ndarray:
    return np.where(y > 0, 1, 0.1)


def tanh(x):
    return np.tanh(x)


def dev_tanh(x):
    return 1 - np.tanh(x) ** 2


class Leaky_Relu(Activation):
    def __init__(self):
        Activation.__init__(self, leaky_relu, dev_leaky_relu)

    def clone(self, perturb_std: float = 0.0):
        clone = Leaky_Relu()
        clone.input_size = self.input_size
        clone.output_size = self.output_size
        clone.frozen = self.frozen
        return clone


class Tanh(Activation):
    def __init__(self):
        Activation.__init__(self, tanh, dev_tanh)

    def clone(self, perturb_std: float = 0.0):
        clone = Tanh()
        clone.input_size = self.input_size
        clone.output_size = self.output_size
        clone.frozen = self.frozen
        return clone
