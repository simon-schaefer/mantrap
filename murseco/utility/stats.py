from abc import abstractmethod
from typing import Union

import json
import numpy as np


class Distribution2D:
    def __init__(self, name):
        self.description = {"name": name}

    @abstractmethod
    def pdf_at(self, x: Union[np.ndarray, float], y: Union[np.ndarray, float]) -> Union[None, np.ndarray]:
        """Get probability density function value at given location(s).
        The x, y coordinates can either be single coordinates or larger matrices, the function is shape containing,
        i.e. it returns probability values with the same shape as x and y (MxN). Since the evaluation of the pdf
        function is based on (x, y) pairs hence x and y have to be in the same shape.
        :argument x: array of x points to evaluate (MxN).
        :argument y: array of y points to evaluate (MxN).
        :return pdf values for given (x,y)-pairs (MxN).
        """
        x, y = np.asarray(x), np.asarray(y)
        assert x.shape == y.shape, "x and y have to be in the same shape to compute pdf-values"
        return None

    def to_json(self, filepath: str):
        """Write summary of given distribution to json file, containing the distribution type, parameters, etc.
        :argument filepath to write to.
        """
        file_object = open(filepath, "w")
        return json.dump(self.description, file_object)

    def summary(self) -> str:
        """Return distribution description as json-style formatted string.
        :return json-formatted string describing distribution.
        """
        return json.dumps(self.description)

    @classmethod
    def from_json(cls, filepath: str):
        with open(filepath, "r") as read_file:
            cls.description = json.load(read_file)


class Gaussian2D(Distribution2D):
    """f(x) = 1 / /sqrt(2*pi )^p * det(Sigma)) * exp(-0.5 * (x - mu)^T * Sigma^(-1) * (x - mu))"""

    def __init__(self, mu: np.ndarray, sigma: np.ndarray):
        super(Gaussian2D, self).__init__("gaussian2d")
        mu, sigma = np.squeeze(mu), np.squeeze(sigma)  # prevent e.g. sigma.shape = (1, 2, 2)
        assert mu.size == 2, "mean vector has to be two dimensional"
        assert sigma.shape == (2, 2), "variance matrix has to be two-by-two"
        assert np.linalg.det(sigma) != 0, "variance matrix has to be invertible"
        assert sigma[0, 1] == sigma[1, 0], "variance matrix has to be symmetric"

        self._K1 = 1 / (2 * np.pi * np.sqrt(np.linalg.det(sigma)))
        self._K2 = -0.5 / (sigma[0, 0] * sigma[1, 1] - sigma[0, 1] ** 2)
        self._mu, self._sigma = mu, sigma

        self.description["mu"] = self._mu.tolist()
        self.description["sigma"] = self._sigma.tolist()

    def pdf_at(self, x: Union[np.ndarray, float], y: Union[np.ndarray, float]) -> Union[None, np.ndarray]:
        super(Gaussian2D, self).pdf_at(x, y)
        dx, dy = x - self._mu[0], y - self._mu[1]
        sx, sy, sxy = self._sigma[0, 0], self._sigma[1, 1], self._sigma[0, 1]
        return self._K1 * np.exp(self._K2 * (dx ** 2 * sy - 2 * sxy * dx * dy + dy ** 2 * sx))

    @classmethod
    def from_json(cls, filepath: str):
        super(Gaussian2D, cls).from_json(filepath)
        cls.description["mu"] = np.asarray(cls.description["mu"])
        cls.description["sigma"] = np.reshape(np.asarray(cls.description["sigma"]), (2, 2))
        return cls(cls.description["mu"], cls.description["sigma"])


class GMM2D(Distribution2D):
    """f(x) = sum_i w_i * Gaussian2D_i(x)"""

    def __init__(self, mus: np.ndarray, sigmas: np.ndarray, weights: np.ndarray):
        super(GMM2D, self).__init__("gmm2d")
        assert len(mus.shape) == 2, "mus must be a stack of two-dimensional vectors"
        assert len(sigmas.shape) == 3, "sigmas must be a stack of 2x2 matrices"
        assert mus.shape[0] == sigmas.shape[0], "length of mus and sigmas array must be equal"
        assert mus.shape[0] == weights.size, "length of gaussians and weights must be equal"

        self.num_modes = mus.shape[0]
        self._gaussians = [Gaussian2D(mus[i, :], sigmas[i, :, :]) for i in range(self.num_modes)]
        self._weights = weights

        self.description["mus"] = mus.tolist()
        self.description["sigmas"] = sigmas.tolist()
        self.description["weights"] = weights.tolist()

    def pdf_at(self, x: Union[np.ndarray, float], y: Union[np.ndarray, float]) -> Union[None, np.ndarray]:
        super(GMM2D, self).pdf_at(x, y)
        weighted_probabilities = np.array([self._weights[i] * g.pdf_at(x, y) for i, g in enumerate(self._gaussians)])
        return np.sum(weighted_probabilities, axis=0) / np.sum(self._weights)

    @classmethod
    def from_json(cls, filepath: str):
        super(GMM2D, cls).from_json(filepath)
        cls.description["mus"] = np.reshape(np.asarray(cls.description["mus"]), (-1, 2))
        cls.description["sigmas"] = np.reshape(np.asarray(cls.description["sigmas"]), (-1, 2, 2))
        cls.description["weights"] = np.asarray(cls.description["weights"])
        return cls(cls.description["mus"], cls.description["sigmas"], cls.description["weights"])
