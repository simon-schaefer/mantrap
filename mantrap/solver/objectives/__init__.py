from mantrap.constants import *
from mantrap.solver.objectives.control_effort import ControlEffortModule
from mantrap.solver.objectives.goal import GoalModule
from mantrap.solver.objectives.acc_interact import InteractionAccelerationModule
from mantrap.solver.objectives.pos_interact import InteractionPositionModule

OBJECTIVES_DICT = {
    OBJECTIVE_CONTROL_EFFORT: ControlEffortModule,
    OBJECTIVE_GOAL: GoalModule,
    OBJECTIVE_INTERACTION_ACC: InteractionAccelerationModule,
    OBJECTIVE_INTERACTION_POS: InteractionPositionModule,
}

OBJECTIVE_MODULES = list(OBJECTIVES_DICT.values())
OBJECTIVES = list(OBJECTIVES_DICT.keys())
