from mantrap.solver.base.trajopt import TrajOptSolver

from mantrap.solver.sgrad import SGradSolver
from mantrap.solver.mc_tree_search import MonteCarloTreeSearch
from mantrap.solver.baselines.orca import ORCASolver
from mantrap.solver.baselines.ignoring import IgnoringSolver

SOLVERS = [SGradSolver, IgnoringSolver, ORCASolver, MonteCarloTreeSearch]
SOLVERS_DICT = {solver.solver_name(): solver for solver in SOLVERS}
