import mantrap.constants

from mantrap.objectives.objective_module import ObjectiveModule

from mantrap.objectives.acc_interact import InteractionAccelerationModule
from mantrap.objectives.control_effort import ControlEffortModule
from mantrap.objectives.goal import GoalModule
from mantrap.objectives.pos_interact import InteractionPositionModule

OBJECTIVES_DICT = {
    mantrap.constants.OBJECTIVE_CONTROL_EFFORT: ControlEffortModule,
    mantrap.constants.OBJECTIVE_GOAL: GoalModule,
    mantrap.constants.OBJECTIVE_INTERACTION_ACC: InteractionAccelerationModule,
    mantrap.constants.OBJECTIVE_INTERACTION_POS: InteractionPositionModule,
}

OBJECTIVE_MODULES = list(OBJECTIVES_DICT.values())
OBJECTIVES = list(OBJECTIVES_DICT.keys())
