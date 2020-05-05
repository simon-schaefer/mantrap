import mantrap.constants

from mantrap.filter.filter_module import FilterModule
from mantrap.filter.euclidean import EuclideanModule
from mantrap.filter.nofilter import NoFilterModule
from mantrap.filter.reachability import ReachabilityModule

FILTER_DICT = {
    mantrap.constants.FILTER_EUCLIDEAN: EuclideanModule,
    mantrap.constants.FILTER_NO_FILTER: NoFilterModule,
    mantrap.constants.FILTER_REACHABILITY: ReachabilityModule,
}

FILTER_MODULES = list(FILTER_DICT.values())
FILTERS = list(FILTER_DICT.keys())
