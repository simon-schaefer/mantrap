from mantrap.constants import *
from mantrap.solver.constraints.max_speed import MaxSpeedModule
from mantrap.solver.constraints.min_distance import MinDistanceModule

CONSTRAINTS_DICT = {
    CONSTRAINT_MAX_SPEED: MaxSpeedModule,
    CONSTRAINT_MIN_DISTANCE: MinDistanceModule,
}

CONSTRAINT_MODULES = [MaxSpeedModule, MinDistanceModule]
CONSTRAINTS = list(CONSTRAINTS_DICT.keys())
