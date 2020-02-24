import logging

import numpy as np
import torch

from mantrap.utility.io import remove_bytes_from_logging

logging.StreamHandler.emit = remove_bytes_from_logging(logging.StreamHandler.emit)
logging.basicConfig(level=logging.INFO, format="[%(asctime)-15s %(filename)-20s %(levelname)-6s] %(message)-s")
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("numpy").setLevel(logging.WARNING)
torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=4)
