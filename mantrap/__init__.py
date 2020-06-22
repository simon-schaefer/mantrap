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


__set_type_defaults()
