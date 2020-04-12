import logging
import sys

import numpy as np
import torch


def remove_bytes_from_logging(fn):
    """Remove weird IPOPT callbacks logging output (byte strings) from log."""
    def remove_bytes(*args):
        if type(args[1]) == logging.LogRecord and type(args[1].msg) == bytes:
            return
        return fn(*args)
    return remove_bytes


logging.StreamHandler.emit = remove_bytes_from_logging(logging.StreamHandler.emit)
logging.basicConfig(level=logging.ERROR, format="[%(asctime)-15s %(filename)-20s %(levelname)-6s] %(message)-s")
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("numpy").setLevel(logging.WARNING)
torch.set_default_dtype(torch.float32)
np.set_printoptions(precision=4)
np.set_printoptions(threshold=sys.maxsize)  # dont collapse arrays while printing
np.seterr(divide='ignore', invalid='ignore')
