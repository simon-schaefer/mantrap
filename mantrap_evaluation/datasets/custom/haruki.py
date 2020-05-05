import typing

import mantrap
import torch

import mantrap_evaluation.datasets.api


def scenario_custom_haruki(
    env_type: mantrap.environment.GraphBasedEnvironment.__class__,
    ego_type: mantrap.agents.DTAgent.__class__ = mantrap.agents.DoubleIntegratorDTAgent,
    num_modes: int = 1
) -> typing.Tuple[mantrap.environment.GraphBasedEnvironment,
                  torch.Tensor,
                  typing.Union[typing.Dict[str, torch.Tensor], None]]:
    """Scenario haruki.

    Three ado agents, having 2 modes each, and the ego agent. All starting at rest, start at both "sides" of the
    environment (left, right) and moving into the opposite direction while crossing each other on the way between.
    """
    ego_state = torch.tensor([-7, 0, 0, 0])
    ego_goal = torch.tensor([7, -1])
    ado_histories = [torch.tensor([-7, -1, 1, 0, 0]),
                     torch.tensor([7, 3, -1, 0, 0]),
                     torch.tensor([7, -2, -1, 1, 0])
                     ]
    ado_histories = [history.unsqueeze(dim=0) for history in ado_histories]  # shape (5) -> (1, 5)
    ado_goals = [torch.tensor([7, -1]), torch.tensor([-7, 1]), torch.tensor([-7, 0])]

    return mantrap_evaluation.datasets.api.create_environment(
        config_name="custom_haruki",
        env_type=env_type,
        ado_histories=ado_histories,
        ego_type=ego_type,
        ego_state=ego_state,
        ado_goals=ado_goals,
        num_modes=num_modes,

    ), ego_goal, None
