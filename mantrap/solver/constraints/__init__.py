from mantrap.constants import *
from mantrap.solver.constraints.max_speed import MaxSpeedModule
from mantrap.solver.constraints.min_distance import MinDistanceModule
from mantrap.solver.constraints.norm_distance import NormDistanceModule

CONSTRAINTS_DICT = {
    CONSTRAINT_MAX_SPEED: MaxSpeedModule,
    CONSTRAINT_MIN_DISTANCE: MinDistanceModule,
    CONSTRAINT_NORM_DISTANCE: NormDistanceModule,
}

CONSTRAINT_MODULES = list(CONSTRAINTS_DICT.values())
CONSTRAINTS = list(CONSTRAINTS_DICT.keys())
