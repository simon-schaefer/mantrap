#######################################
# names/strings #######################
#######################################
ID_EGO = "ego"

LT_GRADIENT = "grad"  # LT = Log-Type
LT_CONSTRAINT = "inf"
LT_OBJECTIVE = "obj"
LT_EGO = "ego"
LT_ADO = "ado"
LT_ADO_WO = "ado_wo"

LK_OVERALL = "overall"  # LK = Log-Keys (module names, ...)

TAG_OPTIMIZATION = "optimization"  # logging tags
TAG_WARM_START = "warm_start"
TAG_VISUALIZATION = "visualization"

PK_CONFIG = "config_name"  # PK = Parameter-Key
PK_GOAL = "goal"
PK_T_PLANNING = "t_planning"
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
ENV_VAR_INITIAL = 1e-6  # variance of initial distribution at t = t0.

ENV_NUM_PARTICLES = 5  # number of particles for estimating velocity distribution for particle based predictions.
ENV_PARTICLE_NOISE = 1e-6  # velocity noise to avoid running into troubles in case of otherwise zero-variance.

KALMAN_ADDITIVE_NOISE = 0.2  # additive noise per prediction time-step (Q in Kalman equations).

SOCIAL_FORCES_DEFAULT_TAU = 0.4  # [s] relaxation time (assumed to be uniform over all agents).
SOCIAL_FORCES_DEFAULT_V0 = 4.0, 1.0  # [m2s-2] repulsive field constant (mean, variance).
SOCIAL_FORCES_DEFAULT_SIGMA = 0.9, 0.3  # [m] repulsive field exponent constant (mean, variance).
SOCIAL_FORCES_MAX_GOAL_DISTANCE = 0.3  # [m] maximal distance to goal to have zero goal traction force.
SOCIAL_FORCES_MAX_INTERACTION_DISTANCE = 2.0  # [m] maximal distance between agents for interaction force.

POTENTIAL_FIELD_V0_DEFAULT = 2.0, 1.0  # [m2s-2] repulsive field constant (mean/variance).
POTENTIAL_FIELD_MAX_THETA = 30.0  # [deg] maximal attention angle to be influenced by robot

TRAJECTRON_MODEL = ("models_18_Jan_2020_01_42_46eth_rob", 1999)  # trajectron model file and iteration number.
TRAJECTRON_DEFAULT_HISTORY_LENGTH = 5

SGAN_MODEL = "models/sgan-models/eth_8_model.pt"

#######################################
# solver parameters ###################
#######################################
SOLVER_HORIZON_DEFAULT = 5  # number of future time-steps to be taken into account
SOLVER_CONSTRAINT_LIMIT = 1e-3  # limit of sum of constraints to be fulfilled
SOLVER_GOAL_END_DISTANCE = 0.5  # [m] maximal distance to goal to finish optimization.

WARM_START_HARD = "hard"  # warm-starting methods
WARM_START_ENCODING = "encoding"
WARM_START_SOFT = "soft"
WARM_START_POTENTIAL = "potential"
WARM_START_ZEROS = "zeros"

WARM_START_PRE_COMPUTATION_NUM = 100  # number of randomly pre-computed scenarios.
WARM_START_PRE_COMPUTATION_HORIZON = 10  # pre-computed time-horizon.
WARM_START_PRE_COMPUTATION_FILE = ("encoding.pt", "solution.pt")

IPOPT_MAX_CPU_TIME_DEFAULT = 2.0  # [s] maximal IPOPT solver CPU time.
IPOPT_OPTIMALITY_TOLERANCE = 0.1  # maximal optimality error to return solution (see IPOPT documentation).
IPOPT_AUTOMATIC_JACOBIAN = "finite-difference-values"  # method for Jacobian approximation (if flag is True).
IPOPT_AUTOMATIC_HESSIAN = "limited-memory"  # method for Hessian approximation.

SEARCH_MAX_CPU_TIME = 0.5  # [s] maximal CPU time of search algorithm.
MCTS_NUMBER_BREADTH_SAMPLES = 5  # number of samples in breadth (i.e. z-values to estimate value from).
MCTS_NUMBER_DEPTH_SAMPLES = 5  # number of samples in depth (i.e. trajectories to estimate value).

RRT_ITERATIONS = 600  # number of sampling iterations (in fact always iteration is looping condition).
RRT_PED_RADIUS = 1.0  # [m] minimal safety distance around pedestrian in RRT-path-planning.
RRT_REWIRE_RADIUS = 10.0  # [m] radius of re-wiring nodes after new node has been added.
RRT_GOAL_SAMPLING_PROBABILITY = 20.0  # probability of sampling and rewiring goal node.

ORCA_AGENT_RADIUS = 0.5  # ado collision-avoidance safety radius [m].
ORCA_MAX_GOAL_DISTANCE = 0.5  # [m] maximal distance to goal to have zero preferred velocity.
ORCA_SAFE_TIME = 0.8  # [s] time interval of guaranteed no collisions. The larger it is, the tighter the
# constraints, but more deviating paths.

OBJECTIVE_PROB_INTERACT_MAX = 30.0  # maximal value of projection probability log cost.
OBJECTIVE_POS_INTERACT_MAX = 3.0  # maximal value of positional-distance cost.
OBJECTIVE_VEL_INTERACT_MAX = 1.0  # maximal value of velocity-distance cost.
OBJECTIVE_ACC_INTERACT_MAX = 1.0  # maximal value of acceleration-distance cost.

CONSTRAINT_HJ_MAT_FILE = "2D.mat"  # pre-computed value-/gradient grid mat file.
CONSTRAINT_HJ_INTERPOLATION_METHOD = "linear"  # value function interpolation method (linear, nearest).
CONSTRAINT_MIN_L2_DISTANCE = 0.5  # [m] minimal distance constraint between ego and every ado ghost
CONSTRAINT_VIOLATION_PRECISION = 1e-5  # allowed precision error when determining constraint violation.

ATTENTION_EUCLIDEAN_RADIUS = 4.0  # [m] attention radius of ego for planning
ATTENTION_CLOSEST_RADIUS = 4.0  # [m] attention radius of ego for planning

#######################################
# visualization parameters ############
#######################################
CONFIG_UNKNOWN = "unknown"
VISUALIZATION_AGENT_RADIUS = 0.1
VISUALIZATION_DIRECTORY = "outputs/"
VISUALIZATION_SAMPLES = 10
VISUALIZATION_FRAME_DELAY = 400  # [ms]
VISUALIZATION_RESTART_DELAY = 2000  # [ms]
