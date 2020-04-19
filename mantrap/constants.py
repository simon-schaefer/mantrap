#######################################
# names/strings #######################
#######################################
CONSTRAINT_MAX_SPEED = "max_speed"
CONSTRAINT_MIN_DISTANCE = "min_distance"
CONSTRAINT_NORM_DISTANCE = "norm_distance"

ID_EGO = "ego"

FILTER_EUCLIDEAN = "euclidean"
FILTER_NO_FILTER = "no_filter"

GK_CONTROL = "control"  # GK = Graph-Key
GK_POSITION = "position"
GK_VELOCITY = "velocity"

LK_GRADIENT = "grad"  # LK = Log-Key
LK_CONSTRAINT = "inf"
LK_OBJECTIVE = "obj"
LK_OPTIMAL = "opt"
LK_OVERALL_PERFORMANCE = "overall"

OBJECTIVE_GOAL = "goal"
OBJECTIVE_INTERACTION = "interaction"
OBJECTIVE_INTERACTION_POS = "interaction_pos"
OBJECTIVE_INTERACTION_ACC = "interaction_acc"

PARAMS_CONFIG = "config_name"
PARAMS_MULTIPROCESSING = "multiprocessing"
PARAMS_NUM_CONTROL_POINTS = "num_control_points"
PARAMS_GOAL = "goal"
PARAMS_TAU = "tau"
PARAMS_T_PLANNING = "t_planning"
PARAMS_VERBOSE = "verbose"
PARAMS_V0 = "v0"
PARAMS_SIGMA = "sigma"
PARAMS_X_AXIS = "x_axis"
PARAMS_Y_AXIS = "y_axis"

#######################################
# agent parameters ####################
#######################################
AGENT_SPEED_MAX = 4.0  # maximal agent velocity in [m/s].
AGENT_ACC_MAX = 2.0  # maximal agent acceleration in [m/s^2].
ROBOT_SPEED_MAX = 2.0  # maximal robot velocity in [m/s].
ROBOT_ACC_MAX = 2.0  # maximal robot acceleration in [m/s^2].

#######################################
# environment parameters ##############
#######################################
ENV_DT_DEFAULT = 0.4
ENV_X_AXIS_DEFAULT = (-10, 10)
ENV_Y_AXIS_DEFAULT = (-10, 10)

ENV_SOCIAL_FORCES_DEFAULTS = {
    PARAMS_TAU: 0.4,  # [s] relaxation time (assumed to be uniform over all agents).
    PARAMS_V0: 4.0,  # [m2s-2] repulsive field constant.
    PARAMS_SIGMA: 0.9,  # [m] repulsive field exponent constant.
}
ENV_SOCIAL_FORCES_MAX_GOAL_DISTANCE = 0.3  # [m] maximal distance to goal to have zero goal traction force.
ENV_SOCIAL_FORCES_MAX_INTERACTION_DISTANCE = 2.0  # [m] maximal distance between agents for interaction force.

ENV_POTENTIAL_FIELD_V0_DEFAULT = 4.0  # [m2s-2] repulsive field constant.

ENV_ORCA_AGENT_RADIUS = 1.0  # ado collision radius [m].
ENV_ORCA_EPS_NUMERIC = 0.0001
ENV_ORCA_SUB_TIME_STEP = 0.01  # [s] interval the simulation time-steps are divided in.
ENV_ORCA_MAX_GOAL_DISTANCE = 0.5  # [m] maximal distance to goal to have zero preferred velocity.
ENV_ORCA_SAFE_TIME = 0.8  # [s] time interval of guaranteed no collisions. The larger it is, the tighter the
# constraints, but more deviating paths.

ENV_TRAJECTRON_MODEL = ("models_18_Jan_2020_01_42_46eth_rob", 1999)  # trajectron model file and iteration number.

#######################################
# solver parameters ###################
#######################################
SOLVER_HORIZON_DEFAULT = 5  # number of future time-steps to be taken into account
SOLVER_CONSTRAINT_LIMIT = 1e-3  # limit of sum of constraints to be fulfilled

IPOPT_MAX_STEPS_DEFAULT = 100  # maximal number of IPOPT solver iterations.
IPOPT_MAX_CPU_TIME_DEFAULT = 1.0  # [s] maximal IPOPT solver CPU time.

CONSTRAINT_MIN_L2_DISTANCE = 0.5  # [m] minimal distance constraint between ego and every ado ghost

FILTER_EUCLIDEAN_RADIUS = 7.0  # [m] attention radius of ego for planning

MCTS_MAX_STEPS = 1000  # maximal number of samples.
MCTS_MAX_CPU_TIME = 1.0  # [s] maximal sampling CPU time.

#######################################
# visualization parameters ############
#######################################
CONFIG_UNKNOWN = "unknown"
TAG_DEFAULT = LK_OPTIMAL
VISUALIZATION_AGENT_RADIUS = 0.1
VISUALIZATION_DIRECTORY = "outputs/"
