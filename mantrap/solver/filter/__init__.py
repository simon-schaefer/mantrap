from mantrap.constants import *
from mantrap.solver.filter.euclidean import EuclideanModule
from mantrap.solver.filter.nofilter import NoFilterModule

FILTER_DICT = {
    FILTER_EUCLIDEAN: EuclideanModule,
    FILTER_NO_FILTER: NoFilterModule,
}

FILTER = [NoFilterModule, EuclideanModule]
