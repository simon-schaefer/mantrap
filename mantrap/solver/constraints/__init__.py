from mantrap.solver.constraints.initial_point import InitialPointModule
from mantrap.solver.constraints.max_speed import MaxSpeedModule
from mantrap.solver.constraints.path_length import PathLengthModule

CONSTRAINTS = {
    "path_length": PathLengthModule,
    "initial_point": InitialPointModule,
    "max_speed": MaxSpeedModule,
}
