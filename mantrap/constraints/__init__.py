import mantrap.constants

from mantrap.constraints import ControlLimitModule
from mantrap.constraints import MinDistanceModule
from mantrap.constraints import NormDistanceModule

CONSTRAINTS_DICT = {
    mantrap.constants.CONSTRAINT_CONTROL_LIMIT: ControlLimitModule,
    mantrap.constants.CONSTRAINT_MIN_DISTANCE: MinDistanceModule,
    mantrap.constants.CONSTRAINT_NORM_DISTANCE: NormDistanceModule,
}

CONSTRAINT_MODULES = list(CONSTRAINTS_DICT.values())
CONSTRAINTS = list(CONSTRAINTS_DICT.keys())
