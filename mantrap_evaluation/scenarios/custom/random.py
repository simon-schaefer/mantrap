import typing

import mantrap
import torch

import mantrap_evaluation.scenarios.api


def random(env_type: mantrap.environment.base.GraphBasedEnvironment.__class__, num_ados: int, **env_kwargs
           ) -> typing.Tuple[mantrap.environment.base.GraphBasedEnvironment,
                             torch.Tensor,
                             typing.Union[typing.Dict[str, torch.Tensor], None]]:
    """Scenario random.

    Create a random scenario with N pedestrians in the scene.
    """
    def random_state(is_robot: bool = False) -> torch.Tensor:
        x_min, x_max = mantrap.constants.ENV_X_AXIS_DEFAULT
        position = torch.rand(2) * (x_max - x_min) + x_min
        if is_robot:
            velocity = torch.zeros(2)
        else:
            velocity = torch.rand(2) * mantrap.constants.PED_SPEED_MAX / 2
        return torch.cat([position, velocity, torch.zeros(1)])

    ego_state = random_state(is_robot=True)
    ego_goal = random_state(is_robot=True)[0:2]
    ado_histories = [random_state(is_robot=False) for _ in range(num_ados)]
    ado_goals = [random_state(is_robot=False)[0:2] for _ in range(num_ados)]
    ado_histories = [history.unsqueeze(dim=0) for history in ado_histories]  # shape (5) -> (1, 5)

    return mantrap_evaluation.scenarios.api.create_environment(
        config_name="random",
        env_type=env_type,
        ado_histories=ado_histories,
        ego_state=ego_state,
        ado_goals=ado_goals,
        **env_kwargs
    ), ego_goal, None
