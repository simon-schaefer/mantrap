from mantrap.constants import *
from mantrap.solver.filter.euclidean import EuclideanModule
from mantrap.solver.filter.nofilter import NoFilterModule
from mantrap.solver.filter.reachability import ReachabilityModule

FILTER_DICT = {
    FILTER_EUCLIDEAN: EuclideanModule,
    FILTER_NO_FILTER: NoFilterModule,
    FILTER_REACHABILITY: ReachabilityModule,
}

FILTER_MODULES = list(FILTER_DICT.values())
FILTERS = list(FILTER_DICT.keys())
