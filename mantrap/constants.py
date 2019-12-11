t_horizon_default = 10
planning_horizon_default = 30

#######################################
# simulation parameters ###############
#######################################
sim_dt_default = 0.2
sim_x_axis_default = (-10, 10)
sim_y_axis_default = (-10, 10)
sim_speed_max = 4

sim_social_forces_tau = 0.5  # [s] relaxation time (assumed to be uniform over all agents).
sim_social_forces_v_0 = 2.1  # [m2s-2] repulsive field constant.
sim_social_forces_sigma = 1.0  # [m] repulsive field exponent constant.
sim_social_forces_min_goal_distance = 0.1  # [m] minimal distance to goal to have non-zero goal traction force

sim_distance_field_sigma = 0.1  # [m] repulsive field exponent constant.

#######################################
# solver parameters ###################
#######################################
igrad_radius = 5.0  # [m] maximal distance for ados to be taken into account

igrad_predictive_horizon = 10  # number of future time-steps to be taken into account

#######################################
# visualization parameters ############
#######################################
visualization_agent_radius = 0.1
visualization_preview_horizon = 10
