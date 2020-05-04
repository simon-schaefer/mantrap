from typing import List

import torch

from mantrap.agents.agent import Agent
from mantrap.environment.environment import GraphBasedEnvironment
from mantrap.utility.shaping import check_ego_state, check_ego_trajectory, check_goal


def _create_environment(
    env_type: GraphBasedEnvironment.__class__,
    config_name: str,
    ado_histories: List[torch.Tensor],
    ego_type: Agent.__class__ = None,
    ego_state: torch.Tensor = None,
    ado_ids: List[str] = None,
    ado_goals: List[torch.Tensor] = None,
    num_modes: int = 1,
    **env_kwargs
) -> GraphBasedEnvironment:
    """Create an environment based on given state and state-histories of all agents in the scene as well as
    several environment properties such as the number of modes and it's type.
    """
    assert all([check_ego_trajectory(ado_history) for ado_history in ado_histories])
    assert ego_state is None or check_ego_state(ego_state, enforce_temporal=False)
    assert ado_goals is None or all([check_goal(goal) for goal in ado_goals])
    assert num_modes >= 1

    ego_kwargs = {"position": ego_state[0:2], "velocity": ego_state[2:4]} if ego_type is not None else None
    env = env_type(ego_type=ego_type, ego_kwargs=ego_kwargs, config_name=config_name, **env_kwargs)
    for m_ado, history in enumerate(ado_histories):
        ado_kwargs = {
            "position": history[-1, 0:2].float(),
            "velocity": history[-1, 2:4].float(),
            "history": history.float(),
            "goal": ado_goals[m_ado].float() if ado_goals is not None else None,
            "identifier": ado_ids[m_ado] if ado_ids is not None else None,
        }
        env.add_ado(num_modes=num_modes, **ado_kwargs)
    return env
