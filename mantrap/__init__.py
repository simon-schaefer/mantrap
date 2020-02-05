import logging

import numpy as np

from mantrap.utility.io import add_coloring_to_ansi

logging.StreamHandler.emit = add_coloring_to_ansi(logging.StreamHandler.emit)
logging.basicConfig(level=logging.INFO, format="[%(asctime)-15s %(filename)-10s %(levelname)-6s] %(message)s")
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("numpy").setLevel(logging.WARNING)
np.set_printoptions(precision=4)
