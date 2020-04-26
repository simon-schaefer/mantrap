from mantrap.solver.sgrad import SGradSolver
from mantrap.solver.mc_tree_search import MonteCarloTreeSearch
from mantrap.solver.solver_baselines.orca import ORCASolver
from mantrap.solver.solver_baselines.ignoring import IgnoringSolver

# SOLVER = [IGradSolver, SGradSolver, ORCASolver, MonteCarloTreeSearch]
SOLVERS = [SGradSolver, IgnoringSolver, ORCASolver, MonteCarloTreeSearch]
