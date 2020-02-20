import importlib
import inspect
import logging
import time
from typing import Callable, List

import numpy as np
import torch

from mantrap.agents import IntegratorDTAgent
from mantrap.constants import agent_speed_max
from mantrap.simulation import PotentialFieldSimulation
from mantrap.solver import SGradSolver, IGradSolver
from mantrap.utility.io import build_os_path
from mantrap.utility.primitives import straight_line


class RunTimeAnalysis:

    def __init__(self, num_agents: List[float], horizons: List[float]):
        self._num_agents = num_agents
        self._horizons = horizons
        self.run_times_dict = {}

    def measure(self, function: Callable) -> np.ndarray:
        run_times = np.zeros((len(self._num_agents), len(self._horizons)))

        for i, num_agents in enumerate(self._num_agents):

            # Define function arguments. Pass every kind of argument that might be required, if not then it wont
            # use the argument anyways, without creating much of an overhead.
            function_kwargs = {}
            sim = PotentialFieldSimulation(IntegratorDTAgent, {"position": torch.tensor([2.0, -5.0])})
            for _ in range(int(num_agents)):
                ado_position = torch.from_numpy(np.random.uniform(-10, 10, size=2))
                ado_velocity = torch.from_numpy(np.random.uniform(-0.5, 0.5, size=2)) * agent_speed_max
                sim.add_ado(position=ado_position, velocity=ado_velocity)
            function_kwargs["sim"] = sim
            function_kwargs["goal"] = torch.tensor([4, 4])
            function_kwargs["num_agents"] = num_agents

            # Measure runtime for different parameter sets.
            for j, horizon in enumerate(self._horizons):
                start_time = time.time()
                function(horizon=horizon, **function_kwargs)
                run_times[i, j] = time.time() - start_time

        # Log run times of this measurement in internal log dictionary.
        self.run_times_dict[function.__name__.replace("measure_", "")] = run_times
        return run_times

    def visualize(self):
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, len(self._horizons) * 3), constrained_layout=True)
        plt.title(f"Run Times Measurement")
        plt.axis("off")
        grid = plt.GridSpec(len(self._horizons), 1, wspace=0.4, hspace=0.3)

        for j, horizon in enumerate(self._horizons):
            ax = fig.add_subplot(grid[j, :])
            ax.set_title(f"Horizon = {horizon}")
            for key, values in self.run_times_dict.items():
                ax.plot(self._num_agents, values[:, j], label=key)
            plt.legend()
            plt.grid()
            ax.set_ylabel("RunTime [s]")

        plt.savefig(build_os_path("test/graphs/runtime_analysis.png", make_dir=False))
        plt.close()


###########################################################################
# Functions ###############################################################
###########################################################################
def measure_igrad_objective(horizon: int, **kwargs):
    solver = IGradSolver(**kwargs, verbose=False, T=horizon)
    x0 = np.random.uniform(-10, 10, size=2)
    solver.objective(x=x0)


def measure_cgrad_objective(horizon: int, **kwargs):
    solver = SGradSolver(**kwargs, verbose=False, T=horizon)
    x0 = straight_line(start_pos=kwargs["sim"].ego.position, end_pos=solver.goal, steps=horizon).detach().numpy()
    solver.objective(x=x0)


def measure_igrad_gradient(horizon: int, **kwargs):
    solver = IGradSolver(**kwargs, verbose=False, T=horizon)
    x0 = np.random.uniform(-10, 10, size=2)
    solver.gradient(x=x0)


def measure_cgrad_gradient(horizon: int, **kwargs):
    solver = SGradSolver(**kwargs, verbose=False, T=horizon)
    x0 = straight_line(start_pos=kwargs["sim"].ego.position, end_pos=solver.goal, steps=horizon).detach().numpy()
    solver.gradient(x=x0)


def measure_sim_prediction(horizon: int, **kwargs):
    kwargs["sim"].predict(t_horizon=horizon)


def measure_sim_connected_graph(horizon: int, **kwargs):
    kwargs["sim"].build_connected_graph(graph_input=torch.ones((horizon, 2)))


if __name__ == '__main__':
    analysis = RunTimeAnalysis(num_agents=[1, 2, 3, 5, 7, 10], horizons=list(range(3, 20, 2)))
    module = importlib.__import__("runtime_analysis")
    measurements = [o for o in inspect.getmembers(module) if inspect.isfunction(o[1]) and "measure" in o[0]]
    for m in measurements:
        logging.info(f"Running measurement ==> {m[0]}")
        analysis.measure(function=m[1])
    analysis.visualize()
