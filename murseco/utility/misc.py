import random
import string

import numpy as np


MATPLOTLIB_COLORS = ["b", "g", "r", "c", "m", "y", "k"]
MATPLOTLIB_MARKERS = ["-", "x", "-.", ":", ".", "o", "v", ">", "<", "1", "2", "3", "4", "s", "p", "*", "h", "+", "d"]


def random_string(length: int = 5) -> str:
    """Generate a random string of fixed given length."""
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


def cardinal_directions() -> np.ndarray:
    """Return cardinal directions as unit vectors in (4, 2) array."""
    return np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
