"""
Environment:
X environment types
X initial conditions
- ego agent type
- simulation time-step

Solver
X solver types
X environment types
X evaluation environment types
X planning horizon: greed (T=1), medium (T=1s), long (T=2s)
- objectives: (goal, interaction), goal only, interaction only
- objectives weight distribution (for non-single permutations)
- constraints: (max_speed, min_distance), max_speed only
X filter: no_filter, uni-modal, attention radius
- IPOPT configuration: automatic Jacobian estimate, manual Jacobian estimate

Specific Comparisons
- control effort objective vs  max control constraint only
"""
import itertools
import os
import time
import typing

import mantrap
import pandas as pd
import tqdm

from mantrap_evaluation import get_metrics, evaluate_metrics
from mantrap_evaluation.utility.modules import load_class_from_module, load_functions_from_module


def configurations() -> typing.Tuple[typing.List, typing.List]:
    configurations_list = [
        load_class_from_module("mantrap.solver", as_tuples=True),  # solver type
        load_functions_from_module("mantrap_evaluation.datasets", as_tuples=True),  # scenario
        load_class_from_module("mantrap.environment", as_tuples=True),  # environment type
        load_class_from_module("mantrap.agents", as_tuples=True),  # ego_type
        load_class_from_module("mantrap.filter", as_tuples=True),  # filter types
        [("greedy", 1), ("medium", 3), ("long", 5)],  # planning horizon
        [("none", None)],  # evaluation environments,
        [("true", True), ("false", False)]  # multi-processing
    ]
    configurations_list = list(itertools.product(*configurations_list))
    config_keys = ["solver", "scenario", "env_type", "ego_type", "filter", "t_planning", "eval_env", "multiprocessing"]
    return configurations_list, config_keys


def run_evaluation(solver, env_original, time_steps: int = 5, **solver_kwargs) -> typing.Dict[str, float]:
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
    configs, config_keys = configurations()
    metrics = list(get_metrics().keys()) + ["runtime"]

    results_df = pd.DataFrame(None, columns=config_keys + metrics)

    print(len(configs))
    for config in tqdm.tqdm(configs):
        config_names, config_objects = zip(*config)
        config_kwargs = dict(zip(config_keys, config_objects))
        env, goal, _ = config_kwargs["scenario"](env_type=config_kwargs["env_type"],
                                                 ego_type=config_kwargs["ego_type"])
        solver = config_kwargs["solver"](env, goal=goal, **config_kwargs)

        try:
            results = run_evaluation(solver,
                                     env_original=env,
                                     multiprocessing=config_kwargs["multiprocessing"]
                                     )
            results.update(config_kwargs)
            results_df = results_df.append(results, ignore_index=True)

            output_path = mantrap.constants.VISUALIZATION_DIRECTORY
            output_path = mantrap.utility.io.build_os_path(os.path.join(output_path, "evaluation.csv"))
            results_df.to_csv(output_path, index=False)

        except:
            print(config_kwargs)


if __name__ == '__main__':
    main()
