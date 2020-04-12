from mantrap.constants import *
from mantrap.solver.filter.euclidean import EuclideanModule
from mantrap.solver.filter.nofilter import NoFilterModule

FILTER = {
    FILTER_EUCLIDEAN: EuclideanModule,
    FILTER_NO_FILTER: NoFilterModule,
}
