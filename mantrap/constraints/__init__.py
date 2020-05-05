import mantrap.constants

from mantrap.constraints.constraint_module import ConstraintModule

from mantrap.constraints.control_limits import ControlLimitModule
from mantrap.constraints.min_distance import MinDistanceModule
from mantrap.constraints.norm_distance import NormDistanceModule

CONSTRAINTS_DICT = {
    mantrap.constants.CONSTRAINT_CONTROL_LIMIT: ControlLimitModule,
    mantrap.constants.CONSTRAINT_MIN_DISTANCE: MinDistanceModule,
    mantrap.constants.CONSTRAINT_NORM_DISTANCE: NormDistanceModule,
}

CONSTRAINT_MODULES = list(CONSTRAINTS_DICT.values())
CONSTRAINTS = list(CONSTRAINTS_DICT.keys())
