import typing

import mantrap
import torch

import mantrap_evaluation.scenarios.api


def custom_surrounding(env_type: mantrap.environment.base.GraphBasedEnvironment.__class__, **env_kwargs
                       ) -> typing.Tuple[mantrap.environment.base.GraphBasedEnvironment,
                                         torch.Tensor,
                                         typing.Union[typing.Dict[str, torch.Tensor], None]]:
    """Scenario surrounding.

    One agent rests in the middle and the robot has to surround it to get to its goal.
    """
    ego_state = torch.tensor([-6, 0, 0, 0])
    ego_goal = torch.tensor([6, 0])
    ado_histories = torch.tensor([0, 0, 0, 0, 0]).unsqueeze(dim=0)
    ado_goals = torch.tensor([0, 0])

    return mantrap_evaluation.scenarios.api.create_environment(
        config_name="custom_surrounding",
        env_type=env_type,
        ado_histories=[ado_histories],
        ego_type=mantrap.agents.DoubleIntegratorDTAgent,
        ego_state=ego_state,
        ado_goals=[ado_goals],
        **env_kwargs
    ), ego_goal, None
