from mantrap.constants import *
from mantrap.solver.filter.euclidean import EuclideanModule
from mantrap.solver.filter.nofilter import NoFilterModule

FILTER_DICT = {
    FILTER_EUCLIDEAN: EuclideanModule,
    FILTER_NO_FILTER: NoFilterModule,
}

FILTER_MODULES = list(FILTER_DICT.values())
FILTERS = list(FILTER_DICT.keys())
