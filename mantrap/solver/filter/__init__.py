from mantrap.solver.filter.euclidean import EuclideanModule
from mantrap.solver.filter.nofilter import NoFilterModule

FILTER = {
    "euclidean": EuclideanModule,
    "none": NoFilterModule,
}
