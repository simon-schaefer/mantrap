from typing import Dict, Tuple, Union

import torch

from mantrap.agents.agent import Agent
from mantrap.agents import DoubleIntegratorDTAgent
from mantrap.environment.environment import GraphBasedEnvironment

from mantrap_evaluation.datasets.api import _create_environment


def scenario_custom_parallel(
    env_type: GraphBasedEnvironment.__class__,
    ego_type: Agent.__class__ = DoubleIntegratorDTAgent,
    num_modes: int = 1
) -> Tuple[GraphBasedEnvironment, torch.Tensor, Union[Dict[str, torch.Tensor], None]]:
    """Scenario parallel.

    Three ado agents, having one mode each, and the ego agent. All starting at the left side and going to the
    right side of the environment, widely independently.
    """
    ego_state = torch.tensor([-7, 0, 0, 0])
    ego_goal = torch.tensor([7, 0])
    ado_histories = [torch.tensor([-7, -5, 1, 0, 0]),
                     torch.tensor([7, 5, 1, 0, 0])
                     ]
    ado_histories = [history.unsqueeze(dim=0) for history in ado_histories]  # shape (5) -> (1, 5)
    ado_goals = [torch.tensor([7, -5]), torch.tensor([7, 5])]

    return _create_environment(
        config_name="custom_haruki",
        env_type=env_type,
        ado_histories=ado_histories,
        ego_type=ego_type,
        ego_state=ego_state,
        ado_goals=ado_goals,
        num_modes=num_modes,

    ), ego_goal, None
