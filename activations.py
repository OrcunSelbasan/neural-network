import numpy as np


class Activation:
    def __call__(self, x: np.ndarray, derivative: bool = False) -> np.ndarray:
        if not derivative:
            return self._normal(x)
        else:
            return self._derivate(x)

    @staticmethod
    def _normal(x: np.ndarray):
        raise NotImplementedError

    @staticmethod
    def _derivate(x: np.ndarray):
        raise NotImplementedError


class Tanh(Activation):
    @staticmethod
    def _normal(x: np.ndarray):
        return np.tanh(x)

    @staticmethod
    def _derivate(x: np.ndarray):
        return 1 - np.tanh(x) ** 2


class Sigmoid(Activation):
    @staticmethod
    def _normal(x: np.ndarray):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _derivate(x: np.ndarray):
        return (1.0 / (1.0 + np.exp(-x))) * (1 - (1.0 / (1.0 + np.exp(-x))))


class ReLU(Activation):
    @staticmethod
    def _normal(x: np.ndarray):
        return np.maximum(0.0, x)

    @staticmethod
    def _derivate(x: np.ndarray):
        return np.where(x > 0, 1.0, 0.0)


class LeakyReLU(Activation):
    @staticmethod
    def _normal(x: np.ndarray):
        return x if x > 0 else 0.01*x 

    @staticmethod
    def _derivate(x: np.ndarray):
        return 0.01 if x < 0 else 1
