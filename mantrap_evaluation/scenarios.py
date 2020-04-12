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
    ego_position, ego_velocity, ego_goal = torch.tensor([-7, 0]), torch.zeros(2), torch.tensor([7, -1])
    ado_positions = torch.stack((torch.tensor([-7, -1]), torch.tensor([7, 3]), torch.tensor([7, -2])))
    ado_goals = torch.stack((torch.tensor([7, -1]), torch.tensor([-7, 1]), torch.tensor([-7, 0])))
    ado_velocities = torch.stack((torch.tensor([1, 0]), torch.tensor([-1, 0]), torch.tensor([-1, 1])))

    return _create_scenario(
        env_class=env_class,
        ego_type=ego_type,
        ego_position=ego_position,
        ego_velocity=ego_velocity,
        ado_positions=ado_positions,
        ado_velocities=ado_velocities,
        ado_goals=ado_goals,
        num_modes=2,
        config_name="haruki"
    ), ego_goal


def scenario_independent(
    env_class: GraphBasedEnvironment.__class__,
    ego_type: Agent.__class__ = DoubleIntegratorDTAgent
) -> Tuple[GraphBasedEnvironment, torch.Tensor]:
    """Scenario independent.

    Three ado agents, having one mode each, and the ego agent. All starting at the left side and going to the
    right side of the environment, widely independently.
    """
    ego_position, ego_velocity, ego_goal = torch.tensor([-7, 0]), torch.zeros(2), torch.tensor([7, 0])
    ado_positions = torch.stack((torch.tensor([-7, -5]), torch.tensor([-7, 5])))
    ado_goals = torch.stack((torch.tensor([7, -5]), torch.tensor([7, 5])))
    ado_velocities = torch.stack((torch.tensor([1, 0]), torch.tensor([1, 0])))

    return _create_scenario(
        env_class=env_class,
        ego_type=ego_type,
        ego_position=ego_position,
        ego_velocity=ego_velocity,
        ado_positions=ado_positions,
        ado_velocities=ado_velocities,
        ado_goals=ado_goals,
        num_modes=1,
        config_name="independent"
    ), ego_goal


###########################################################################
# Scene Creator ###########################################################
###########################################################################
def _create_scenario(
    env_class: GraphBasedEnvironment.__class__,
    ego_type: Agent.__class__,
    ego_position: torch.Tensor,
    ego_velocity: torch.Tensor,
    ado_positions: torch.Tensor,
    ado_velocities: torch.Tensor,
    ado_goals: torch.Tensor,
    num_modes: int,
    config_name: str
) -> GraphBasedEnvironment:
    ego_kwargs = {"position": ego_position, "velocity": ego_velocity} if ego_type is not None else None
    env = env_class(ego_type=ego_type, ego_kwargs=ego_kwargs, config_name=config_name)
    for position, ado_goal, velocity in zip(ado_positions, ado_goals, ado_velocities):
        env.add_ado(position=position, goal=ado_goal, velocity=velocity, num_modes=num_modes)
    return env
