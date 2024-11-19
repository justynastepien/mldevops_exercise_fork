import numpy as np
from typing import Tuple

from collections.abc import Callable
import wandb


def activation_function(x: float) -> float:
    return 1 / (1 + np.e ** (-x))


def activation_function_deriv(x: float) -> float:
    return activation_function(x) * (1 - activation_function(x))


class Neuron:
    def __init__(self, input_size: int, act_func: Callable, act_func_deriv: Callable):
        # TODO
        self._init_weights_and_bias(input_size)
        self._activation_function = act_func
        self._activation_function_deriv = act_func_deriv

    def _init_weights_and_bias(self, input_size: int):
        # TODO
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()

    def __call__(self, x: np.array) -> float:
        return self._forward_propagation(x)

    def _forward_propagation(self, x: np.array) -> float:
        # Remember to call self._activation_function(x)
        z = np.dot(self.weights, x) + self.bias
        return self._activation_function(z)

    def gradient_descent(self, x: np.array, y_target: np.array, alpha: float, iterations: int) -> None:
        for _ in range(iterations):
            dw, db = self._backward_propagation(x, y_target)
            self._update_weights_and_bias(dw, db, alpha)
        pass

    def _backward_propagation(self, x: np.array, y: np.array) -> tuple[np.array, np.array]:
        # Return: weights and bias
        y_pred = self._forward_propagation(x)
        error = y_pred - y

        dw = error * self._activation_function_deriv(np.dot(self.weights, x) + self.bias) * x
        db = error * self._activation_function_deriv(np.dot(self.weights, x) + self.bias)
        return dw, db

    def _update_weights_and_bias(self, dw: np.array, db: float, alpha: float):
        self.weights -= alpha * dw
        self.bias -= alpha * db


class NeuralNetwork:
    def __init__(self, input_size: int, act_func: Callable, act_func_deriv: Callable):
        # TODO
        self._neuron_1 = Neuron(input_size, act_func, act_func_deriv)
        self._neuron_2 = Neuron(input_size, act_func, act_func_deriv)
        self._neuron_3 = Neuron(input_size, act_func, act_func_deriv)

    def __call__(self, x: np.array) -> float:
        return self._network_forward_propagation(x)

    def _network_forward_propagation(self, x: np.array) -> float:
        input_3_1 = self._neuron_1(x)
        input_3_2 = self._neuron_2(x)
        input_3 = np.array([input_3_1, input_3_2])
        return self._neuron_3(input_3)

    def _network_backwards_propagation(self, x: np.array, y: np.array) -> None:
        # TODO
        input_3_1 = self._neuron_1(x)
        input_3_2 = self._neuron_2(x)
        input_3 = np.array([input_3_1, input_3_2])
        y_pred = self._neuron_3(input_3)

        output_error = y_pred - y

        self._neuron_3.gradient_descent(input_3, y, 0.1, 1)

        delta_3 = output_error * activation_function_deriv(
            np.dot(self._neuron_3.weights, input_3) + self._neuron_3.bias
        )

        error_1 = delta_3 * self._neuron_3.weights[0]
        error_2 = delta_3 * self._neuron_3.weights[1]

        self._neuron_1.gradient_descent(x, input_3_1 - error_1, 0.1, 1)
        self._neuron_2.gradient_descent(x, input_3_2 - error_2, 0.1, 1)

    def gradient_descent(self, x: np.array, y: np.array) -> None:
        # for _ in range(iterations):
        for xi, yi in zip(x, y):
            self._network_backwards_propagation(xi, yi)


dataset_xor_x = ((0, 0), (0, 1), (1, 0), (1, 1))
dataset_xor_y = (0, 1, 1, 0)

# TODO
network = NeuralNetwork(input_size=2, act_func=activation_function, act_func_deriv=activation_function_deriv)

# Training parameters
alpha = 0.1
iterations = 10000

run = wandb.init(
    # Set the project where this run will be logged
    project="my-awesome-project",
    # Track hyperparameters and run metadata
    config={
        "alpha": alpha,
        "iterations": iterations,
    },
)

for _ in range(iterations):
    for x, y in zip(dataset_xor_x, dataset_xor_y):
        network.gradient_descent(np.array([x]), np.array([y]))
    good = 0
    for i, x in enumerate(dataset_xor_x):
        pred = network(x)
        if pred > 0.5:
            pred_y = 1
        else:
            pred_y = 0
        if pred_y == dataset_xor_y[i]:
            good += 1

    acc = float(good) / 4.0
    print(acc)
    wandb.log({"accuracy": acc})


print("Testing ANN outputs on XOR dataset:")
for x in dataset_xor_x:
    pred = network(x)
    if pred > 0.5:
        print(f"Input: {x}, Predicted Output: {1}")
    else:
        print(f"Input: {x}, Predicted Output: {0}")
