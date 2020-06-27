import typing

import mantrap
import torch

import mantrap_evaluation.scenarios.api


def custom_avoid(env_type: mantrap.environment.base.GraphBasedEnvironment.__class__, **env_kwargs
                 ) -> typing.Tuple[mantrap.environment.base.GraphBasedEnvironment,
                                   torch.Tensor,
                                   typing.Union[typing.Dict[str, torch.Tensor], None]]:
    """Scenario avoid.

    Crowded scenario with 4 agents which have to be avoided in parallel.
    """
    ego_state = torch.tensor([0, 0, 0, 0])
    ego_goal = torch.tensor([8, 0])
    ado_histories = [torch.tensor([2.0, 2.0, 0.0, -0.5, 0]),
                     torch.tensor([-4.0, 2.0, 1.0, 1.0, 0]),
                     torch.tensor([0.0, 5.0, 1.0, -0.7, 0]),
                     torch.tensor([8.0, -5.0, -3.0, 0.1, 0])]
    ado_goals = [torch.tensor([2, -1]),
                 torch.tensor([2.0, 8.0]),
                 torch.tensor([6.0, 0.8]),
                 torch.tensor([-10.0, -4.4])]
    ado_histories = [history.unsqueeze(dim=0) for history in ado_histories]  # shape (5) -> (1, 5)

    return mantrap_evaluation.scenarios.api.create_environment(
        config_name="custom_avoid",
        env_type=env_type,
        ado_histories=ado_histories,
        ego_type=mantrap.agents.DoubleIntegratorDTAgent,
        ego_state=ego_state,
        ado_goals=ado_goals,
        **env_kwargs
    ), ego_goal, None
