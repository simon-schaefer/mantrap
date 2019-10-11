from abc import abstractmethod
from typing import Any, Dict, Union

import numpy as np

from murseco.utility.io import JSONSerializer


class Distribution2D(JSONSerializer):
    def __init__(self, **kwargs):
        kwargs.update({"is_unique": False})
        super(Distribution2D, self).__init__(**kwargs)

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

    @abstractmethod
    def sample(self, num_samples: int) -> np.ndarray:
        """Sample N = num_samples from distribution and return results stacked in array.

        :argument num_samples: number of samples to return.
        :return 2D samples array (num_samples x 2).
        """
        pass

    @abstractmethod
    def summary(self) -> Dict[str, Any]:
        summary = super(Distribution2D, self).summary()
        return summary

    @classmethod
    def from_summary(cls, json_text: Dict[str, Any]):
        summary = super(Distribution2D, cls).from_summary(json_text)
        return summary


class Point2D(Distribution2D):
    """f(x) = direc_delta(x) with x = point (direc_delta is modelled with 100)"""

    def __init__(self, position: np.ndarray, **kwargs):
        kwargs.update({"name": "utility/stats/Point2D"})
        super(Point2D, self).__init__(**kwargs)

        assert position.size == 2, "position must be two-dimensional"
        self.x, self.y = position.tolist()

    def pdf_at(self, x: Union[np.ndarray, float], y: Union[np.ndarray, float]) -> Union[None, np.ndarray]:
        super(Point2D, self).pdf_at(x, y)
        mask = np.logical_and(np.isclose(self.x, x, atol=0.1), np.isclose(self.y, y, atol=0.1)).astype(int)
        if np.amax(mask) == 0:
            return np.zeros_like(x)
        else:
            return mask * 100  # np.inf

    def sample(self, num_samples: int) -> np.ndarray:
        return np.reshape(np.array([self.x, self.y] * num_samples), (num_samples, 2))

    def summary(self) -> Dict[str, Any]:
        summary = super(Point2D, self).summary()
        summary.update({"position": [self.x, self.y]})
        return summary

    @classmethod
    def from_summary(cls, json_text: Dict[str, Any]):
        summary = super(Point2D, cls).from_summary(json_text)
        summary.update({"position": np.asarray(json_text["position"])})
        return cls(**summary)


class Gaussian2D(Distribution2D):
    """f(x) = 1 / /sqrt(2*pi )^p * det(Sigma)) * exp(-0.5 * (x - mu)^T * Sigma^(-1) * (x - mu))"""

    def __init__(self, mu: np.ndarray, sigma: np.ndarray, **kwargs):
        kwargs.update({"name": "utility/stats/Gaussian2D"})
        super(Gaussian2D, self).__init__(**kwargs)
        mu, sigma = np.squeeze(mu), np.squeeze(sigma)  # prevent e.g. sigma.shape = (1, 2, 2)

        assert mu.size == 2, "mean vector has to be two dimensional"
        assert sigma.shape == (2, 2), "variance matrix has to be two-by-two"
        assert np.linalg.det(sigma) != 0, "variance matrix has to be invertible"
        assert sigma[0, 1] == sigma[1, 0], "variance matrix has to be symmetric"

        self._K1 = 1 / (2 * np.pi * np.sqrt(np.linalg.det(sigma)))
        self._K2 = -0.5 / (sigma[0, 0] * sigma[1, 1] - sigma[0, 1] ** 2)
        self.mu, self.sigma = mu, sigma

    def pdf_at(self, x: Union[np.ndarray, float], y: Union[np.ndarray, float]) -> Union[None, np.ndarray]:
        super(Gaussian2D, self).pdf_at(x, y)
        dx, dy = x - self.mu[0], y - self.mu[1]
        sx, sy, sxy = self.sigma[0, 0], self.sigma[1, 1], self.sigma[0, 1]
        return self._K1 * np.exp(self._K2 * (dx ** 2 * sy - 2 * sxy * dx * dy + dy ** 2 * sx))

    def sample(self, num_samples: int) -> np.ndarray:
        super(Gaussian2D, self).sample(num_samples)
        return np.random.multivariate_normal(self.mu, self.sigma, size=num_samples)

    def summary(self) -> Dict[str, Any]:
        summary = super(Gaussian2D, self).summary()
        summary.update({"mu": self.mu.tolist(), "sigma": self.sigma.tolist()})
        return summary

    @classmethod
    def from_summary(cls, json_text: Dict[str, Any]):
        summary = super(Gaussian2D, cls).from_summary(json_text)
        mu = np.asarray(json_text["mu"])
        sigma = np.reshape(np.asarray(json_text["sigma"]), (2, 2))
        summary.update({"mu": mu, "sigma": sigma})
        return cls(**summary)


class GMM2D(Distribution2D):
    """f(x) = sum_i w_i * Gaussian2D_i(x)"""

    def __init__(self, mus: np.ndarray, sigmas: np.ndarray, weights: np.ndarray, **kwargs):
        kwargs.update({"name": "utility/stats/GMM2D"})
        super(GMM2D, self).__init__(**kwargs)

        mus = mus.squeeze() if len(mus.shape) > 2 else mus  # prevent e.g. mus.shape = (1, 4, 2) but (1, 2)
        sigmas = sigmas.squeeze() if len(sigmas.shape) > 3 else sigmas
        weights = weights.squeeze() if len(weights.shape) > 1 else weights

        assert len(mus.shape) == 2, "mus must be a stack of two-dimensional vectors"
        assert len(sigmas.shape) == 3, "sigmas must be a stack of 2x2 matrices"
        assert mus.shape[0] == sigmas.shape[0], "length of mus and sigmas array must be equal"
        assert mus.shape[0] == weights.size, "length of gaussians and weights must be equal"

        self.num_modes = mus.shape[0]
        self.gaussians = [Gaussian2D(mus[i, :], sigmas[i, :, :]) for i in range(self.num_modes)]
        self.weights = weights / np.sum(weights)  # from now on GMM is normalized
        self.weights = np.round(self.weights, 5)  # prevent testing exact comparison problems

    def pdf_at(self, x: Union[np.ndarray, float], y: Union[np.ndarray, float]) -> Union[None, np.ndarray]:
        super(GMM2D, self).pdf_at(x, y)
        weighted_probabilities = np.array([self.weights[i] * g.pdf_at(x, y) for i, g in enumerate(self.gaussians)])
        return np.sum(weighted_probabilities, axis=0)

    def sample(self, num_samples: int) -> np.ndarray:
        mode_choices = np.random.choice(range(self.num_modes), size=num_samples, p=self.weights)
        return np.array([self.gaussians[mode].sample(1) for mode in mode_choices]).squeeze()

    def summary(self) -> Dict[str, Any]:
        summary = super(GMM2D, self).summary()
        summary.update({"gaussians": [g.summary() for g in self.gaussians], "weights": self.weights.tolist()})
        return summary

    @classmethod
    def from_summary(cls, json_text: Dict[str, Any]):
        summary = super(GMM2D, cls).from_summary(json_text)
        gaussians = [Gaussian2D.from_summary(g) for g in json_text["gaussians"]]
        mus = np.reshape(np.array([g.mu for g in gaussians]), (-1, 2))
        sigmas = np.reshape(np.array([g.sigma for g in gaussians]), (-1, 2, 2))
        weights = np.array(json_text["weights"])
        summary.update({"mus": mus, "sigmas": sigmas, "weights": weights})
        return cls(**summary)
