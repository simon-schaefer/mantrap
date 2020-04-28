from mantrap.constants import *
from mantrap.solver.constraints.control_limits import ControlLimitModule
from mantrap.solver.constraints.min_distance import MinDistanceModule
from mantrap.solver.constraints.norm_distance import NormDistanceModule

CONSTRAINTS_DICT = {
    CONSTRAINT_CONTROL_LIMIT: ControlLimitModule,
    CONSTRAINT_MIN_DISTANCE: MinDistanceModule,
    CONSTRAINT_NORM_DISTANCE: NormDistanceModule,
}

CONSTRAINT_MODULES = list(CONSTRAINTS_DICT.values())
CONSTRAINTS = list(CONSTRAINTS_DICT.keys())
