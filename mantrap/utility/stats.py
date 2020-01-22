from abc import abstractmethod

import numpy as np


class Distribution:
    def __init__(self, mean: float):
        self.mean = mean

    @abstractmethod
    def sample(self, num_samples: int = 1) -> np.ndarray:
        """Sample N = num_samples from distribution and return results stacked in array.

        :argument num_samples: number of samples to return.
        :return samples in array (num_samples, n).
        """
        pass


class DirecDelta(Distribution):
    def __init__(self, x: float):
        super(DirecDelta, self).__init__(x)

    def sample(self, num_samples: int = 1) -> np.ndarray:
        super(DirecDelta, self).sample(num_samples)
        return np.tile(self.mean, num_samples).reshape(num_samples, -1)


class Gaussian(Distribution):
    def __init__(self, mean: float, sigma: float):
        super(Gaussian, self).__init__(mean)
        self.sigma = sigma

    def sample(self, num_samples: int = 1) -> np.ndarray:
        super(Gaussian, self).sample(num_samples)
        return np.random.normal(self.mean, self.sigma, num_samples).reshape(num_samples, -1)
