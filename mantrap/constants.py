#######################################
# names/strings #######################
#######################################
ID_EGO = "ego"

GK_CONTROL = "control"  # GK = Graph-Key
GK_POSITION = "position"
GK_VELOCITY = "velocity"

LT_GRADIENT = "grad"  # LT = Log-Type
LT_CONSTRAINT = "inf"
LT_OBJECTIVE = "obj"

LK_OVERALL_PERFORMANCE = "overall" # LK = Log-Keys (module names, ...)

TAG_OPTIMIZATION = "optimization"  # logging tags
TAG_WARM_START = "warm_start"
TAG_VISUALIZATION = "visualization"

PK_CONFIG = "config_name"  # PK = Parameter-Key
PK_NUM_CONTROL_POINTS = "num_control_points"
PK_GOAL = "goal"
PK_TAU = "tau"
PK_T_PLANNING = "t_planning"
PK_OPTIMIZE_SPEED = "optimize_speed"
PK_V0 = "v0"
PK_SIGMA = "sigma"
PK_X_AXIS = "x_axis"
PK_Y_AXIS = "y_axis"

#######################################
# agent parameters ####################
#######################################
PED_SPEED_MAX = 2.5  # maximal agent velocity in [m/s] according to "Comfortable and maximum walking speed of
# adults aged 20-79 years: reference values and determinants"
PED_ACC_MAX = 2.0  # maximal agent acceleration in [m/s^2].
ROBOT_SPEED_MAX = 2.0  # maximal robot velocity in [m/s].
ROBOT_ACC_MAX = 2.0  # maximal robot acceleration in [m/s^2].

AGENT_MAX_PRE_COMPUTATION = 20  # maximal number of pre-computed time-steps for rolling
# batched dynamics for linear agents (during agent initialization).

#######################################
# environment parameters ##############
#######################################
ENV_DT_DEFAULT = 0.4
ENV_X_AXIS_DEFAULT = (-10, 10)
ENV_Y_AXIS_DEFAULT = (-10, 10)

SOCIAL_FORCES_DEFAULT_TAU = 0.4  # [s] relaxation time (assumed to be uniform over all agents).
SOCIAL_FORCES_DEFAULT_V0 = 4.0  # [m2s-2] repulsive field constant.
SOCIAL_FORCES_DEFAULT_SIGMA = 0.9  # [m] repulsive field exponent constant.
SOCIAL_FORCES_MAX_GOAL_DISTANCE = 0.3  # [m] maximal distance to goal to have zero goal traction force.
SOCIAL_FORCES_MAX_INTERACTION_DISTANCE = 2.0  # [m] maximal distance between agents for interaction force.

POTENTIAL_FIELD_V0_DEFAULT = 4.0  # [m2s-2] repulsive field constant.

ORCA_AGENT_RADIUS = 1.0  # ado collision radius [m].
ORCA_EPS_NUMERIC = 0.0001
ORCA_SUB_TIME_STEP = 0.01  # [s] interval the simulation time-steps are divided in.
ORCA_MAX_GOAL_DISTANCE = 0.5  # [m] maximal distance to goal to have zero preferred velocity.
ORCA_SAFE_TIME = 0.8  # [s] time interval of guaranteed no collisions. The larger it is, the tighter the
# constraints, but more deviating paths.

TRAJECTRON_MODEL = ("models_18_Jan_2020_01_42_46eth_rob", 1999)  # trajectron model file and iteration number.
TRAJECTRON_DEFAULT_HISTORY_LENGTH = 5

#######################################
# solver parameters ###################
#######################################
SOLVER_HORIZON_DEFAULT = 5  # number of future time-steps to be taken into account
SOLVER_CONSTRAINT_LIMIT = 1e-3  # limit of sum of constraints to be fulfilled
SOLVER_GOAL_END_DISTANCE = 0.5  # [m] maximal distance to goal to finish optimization.

IPOPT_MAX_CPU_TIME_DEFAULT = 3.0  # [s] maximal IPOPT solver CPU time.
IPOPT_OPTIMALITY_TOLERANCE = 0.01  # maximal optimality error to return solution (see documentation).
IPOPT_AUTOMATIC_JACOBIAN = "finite-difference-values"  # method for Jacobian approximation (if flag is True).
IPOPT_AUTOMATIC_HESSIAN = "limited-memory"  # method for Hessian approximation.

SEARCH_MAX_CPU_TIME = 1.0  # [s] maximal sampling CPU time.
MCTS_NUMBER_BREADTH_SAMPLES = 5  # number of samples in breadth (i.e. z-values to estimate value from).
MCTS_NUMBER_DEPTH_SAMPLES = 5  # number of samples in depth (i.e. trajectories to estimate value).

CONSTRAINT_MIN_L2_DISTANCE = 0.5  # [m] minimal distance constraint between ego and every ado ghost
CONSTRAINT_VIOLATION_PRECISION = 1e-5  # allowed precision error when determining constraint violation.

FILTER_EUCLIDEAN_RADIUS = 7.0  # [m] attention radius of ego for planning

#######################################
# visualization parameters ############
#######################################
CONFIG_UNKNOWN = "unknown"
VISUALIZATION_AGENT_RADIUS = 0.1
VISUALIZATION_DIRECTORY = "outputs/"
VISUALIZATION_FRAME_DELAY = 400  # [ms]
VISUALIZATION_RESTART_DELAY = 2000  # [ms]

COLORS = [[1, 0, 0], [0, 0, 1], [0.5, 0.5, 0.5], [0.0, 0.8, 0.4], [0.2, 0.6, 0.1], [0.4, 0.1, 0.8],
          [0.5, 0.0, 1.0], [0.6, 1.0, 0.0], [0.4, 0.7, 0.7], [1, 0, 1]]
