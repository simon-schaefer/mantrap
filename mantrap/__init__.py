import mantrap.constants
import mantrap.utility

import mantrap.agents
import mantrap.environment

import mantrap.attention
import mantrap.modules

import mantrap.solver
import mantrap.visualization


#######################################
# Default tensor precision ############
#######################################
def __set_type_defaults():
    import torch
    torch.set_default_dtype(torch.float32)


def __set_seed():
    import numpy as np
    import torch
    np.random.seed(0)
    torch.manual_seed(0)


#######################################
# Logging preferences #################
#######################################
def __set_logging_preferences():
    import logging
    import sys

    import numpy
    import torch

    def remove_bytes_from_logging(fn):
        """Remove weird IPOPT callbacks logging output (byte strings) from log."""

        def remove_bytes(*args):
            if type(args[1]) == logging.LogRecord and type(args[1].msg) == bytes:
                return
            return fn(*args)

        return remove_bytes

    import mantrap.utility.io
    is_debug = __debug__ is True and not mantrap.utility.io.is_running_from_ipython()
    logging.StreamHandler.emit = remove_bytes_from_logging(logging.StreamHandler.emit)
    logging.basicConfig(
        level=logging.DEBUG if is_debug else logging.WARNING,
        format="[%(levelname)-6s > %(filename)-10s:%(lineno)4d (%(asctime)-8s:%(msecs)03d)] %(message)-s",
        datefmt="%H:%M:%S"
    )
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("numpy").setLevel(logging.WARNING)
    torch.set_printoptions(precision=4)
    numpy.set_printoptions(precision=4)
    numpy.set_printoptions(threshold=sys.maxsize)  # dont collapse arrays while printing
    numpy.seterr(divide='ignore', invalid='ignore')


__set_type_defaults()
__set_logging_preferences()
__set_seed()
