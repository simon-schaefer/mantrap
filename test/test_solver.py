from pprint import pprint
import logging

from mantrap.agents import DoubleIntegratorDTAgent, IntegratorDTAgent
from mantrap.evaluation.baselines import straight_line
from mantrap.simulation import SocialForcesSimulation
from mantrap.solver import IGradGreedySolver, IGradPredictiveSolver
from mantrap.evaluation import evaluate, eval_scenarios, scenarios


###########################################################################
# Visualization ###########################################################
###########################################################################
def visualize_igrads():
    solvers = [IGradGreedySolver, IGradPredictiveSolver]
    results = {}
    for solver_class in solvers:
        results[solver_class.__name__] = {}

        for tag, scenario in eval_scenarios().items():
            logging.warning(f"Solver: {solver_class.__name__} -> Test scenario: {tag}")
            test_name = f"{solver_class.__name__}_{tag}"

            sim, goal = scenario(sim_type=SocialForcesSimulation, ego_type=IntegratorDTAgent)
            solver = solver_class(sim, goal=goal)
            ego_traj, ados_traj = solver.solve()

            results[solver_class.__name__][tag], _, _, _ = evaluate(
                test_name, ego_traj, ados_traj, sim, goal, straight_line, do_visualization=True,
            )

    logging.warning("Results summary: ")
    pprint(results)


def visualize_igrad_testing():
    scenario_func = scenarios.scenario_sf_ego_moving_many_ados
    sim, goal = scenario_func(sim_type=SocialForcesSimulation, ego_type=IntegratorDTAgent)
    solver = IGradPredictiveSolver(sim, goal=goal)
    ego_traj, ados_traj = solver.solve()

    test_name = f"{IGradPredictiveSolver.__name__}_{scenario_func.__name__}"
    evaluate(test_name, ego_traj, ados_traj, sim, goal, straight_line, do_visualization=True)


# def visualize_interactive_grad_vector_field():
#     sim, goal = scenarios.scenario_sf_ego_moving_many_ados(ego_type=DoubleIntegratorDTAgent)
#     x_axis, y_axis = sim.axes
#     num_points = 50
#     x_grid, y_grid = np.meshgrid(
#         np.linspace(x_axis[0], x_axis[1], num_points), np.linspace(y_axis[0], y_axis[1], num_points)
#     )
#     vector_field = np.zeros((num_points, num_points, 4))
#     for ix in range(num_points):
#         for iy in range(num_points):
#             vector_field[ix, iy, 0:2] = np.array([x_grid[ix, iy], y_grid[ix, iy]])
#             vector_field[ix, iy, 2:4] = IGradGreedySolver.compute_interaction_grad(sim, vector_field[ix, iy, 0:2])
#             vector_field[ix, iy, 2:4] = -vector_field[ix, iy, 2:4]
#     vector_field = np.reshape(vector_field, (num_points ** 2, 4))
#
#     for k in range(num_points * num_points):
#         vector_length = np.linalg.norm(vector_field[k, 2:4])
#         vector_field[k, 2:4] = vector_field[k, 2:4] / vector_length * min(vector_length, 1)
#
#     import matplotlib.pyplot as plt
#
#     fig, ax = plt.subplots()
#     plt.quiver(x_grid, y_grid, vector_field[:, 2], vector_field[:, 3], units="xy", scale=2)
#     ax.grid(which="minor", alpha=0.1)
#     ax.grid(which="major", alpha=0.3)
#     for ado in sim.ados:
#         print(ado.position)
#     plt.show()
