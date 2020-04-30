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


#######################################
# Default tensor precision ############
#######################################
torch.set_default_dtype(torch.float32)

#######################################
# Logging preferences #################
#######################################
logging.StreamHandler.emit = remove_bytes_from_logging(logging.StreamHandler.emit)
logging.basicConfig(level=logging.WARNING,
                    format="[%(asctime)-8s:%(msecs)03d %(levelname)-6s] %(message)-s",
                    datefmt="%H:%M:%S")
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("numpy").setLevel(logging.WARNING)
torch.set_printoptions(precision=4)
np.set_printoptions(precision=4)
np.set_printoptions(threshold=sys.maxsize)  # dont collapse arrays while printing
np.seterr(divide='ignore', invalid='ignore')
