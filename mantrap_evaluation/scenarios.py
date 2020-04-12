from typing import Tuple

import torch

from mantrap.agents.agent import Agent
from mantrap.agents import DoubleIntegratorDTAgent
from mantrap.environment.environment import GraphBasedEnvironment


def scenario_haruki(
    env_class: GraphBasedEnvironment.__class__,
    ego_type: Agent.__class__ = DoubleIntegratorDTAgent
) -> Tuple[GraphBasedEnvironment, torch.Tensor]:
    """Scenario haruki.

    Three ado agents, having 2 modes each, and the ego agent. All starting at rest, start at both "sides" of the
    environment (left, right) and moving into the opposite direction while crossing each other on the way between.
    """
    ego_position = torch.tensor([-7, 0])
    ego_velocity = torch.zeros(2)
    ego_goal = torch.tensor([7, -1])
    ado_positions = torch.stack((torch.tensor([-7, -1]), torch.tensor([7, 3]), torch.tensor([7, -2])))
    ado_goals = torch.stack((torch.tensor([7, -1]), torch.tensor([-7, 1]), torch.tensor([-7, 0])))
    ado_velocities = torch.stack((torch.tensor([1, 0]), torch.tensor([-1, 0]), torch.tensor([-1, 1])))

    ego_kwargs = {"position": ego_position, "velocity": ego_velocity} if ego_type is not None else None
    env = env_class(ego_type=ego_type, ego_kwargs=ego_kwargs, config_name="haruki")
    for position, ado_goal, velocity in zip(ado_positions, ado_goals, ado_velocities):
        env.add_ado(position=position, goal=ado_goal, velocity=velocity, num_modes=2)
    return env, ego_goal
