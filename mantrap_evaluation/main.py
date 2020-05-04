import os
import time
from typing import Dict

import pandas as pd
import tqdm

from mantrap.constants import *
from mantrap.agents import AGENTS_DICT
from mantrap.environment import ENVIRONMENTS_DICT
from mantrap.solver import SOLVERS_DICT
from mantrap.utility.io import build_os_path

from mantrap_evaluation import get_metrics, evaluate_metrics
from mantrap_evaluation.datasets import SCENARIOS
from mantrap_evaluation.configurations import configurations, config_keys


def run_evaluation(solver, env_original, time_steps: int = 5, **solver_kwargs) -> Dict[str, float]:
    start_time = time.time()

    # Run experiment, i.e. solve for ego and ado trajectories for N time-steps.
    ego_trajectory, ado_trajectories = solver.solve(time_steps=time_steps, **solver_kwargs)

    # Evaluate the solver's performance based on the metrics defined in `mantrap_evaluation.metrics`.
    evaluation_dict = {"runtime": (time.time() - start_time) / time_steps}
    evaluation_dict.update(evaluate_metrics(ego_trajectory=ego_trajectory,
                                            env=env_original,
                                            ado_trajectories=ado_trajectories,
                                            goal=solver.goal
                                            ))
    # Round results to improve their readability.
    evaluation_dict = {key: round(value, 3) for key, value in evaluation_dict.items()}
    return evaluation_dict


def main():
    metrics = list(get_metrics().keys()) + ["runtime"]

    results_df = pd.DataFrame(None, columns=config_keys + metrics)
    for config in tqdm.tqdm(configurations):
        config_kwargs = dict(zip(config_keys, config))

        env_type = ENVIRONMENTS_DICT[config_kwargs["env_type"]]
        ego_type = AGENTS_DICT[config_kwargs["ego_type"]]
        env, goal, _ = SCENARIOS[config_kwargs["scenario"]](env_type=env_type, ego_type=ego_type)
        solver = SOLVERS_DICT[config_kwargs["solver"]](env, goal=goal, **config_kwargs)

        try:
            results = run_evaluation(solver,
                                     env_original=env,
                                     multiprocessing=config_kwargs["multiprocessing"]
                                     )
            results.update(config_kwargs)
            results_df = results_df.append(results, ignore_index=True)

        except:
            print(config_kwargs)

    output_path = build_os_path(os.path.join(VISUALIZATION_DIRECTORY, "evaluation.csv"))
    results_df.to_csv(output_path)


if __name__ == '__main__':
    main()
