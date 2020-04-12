from mantrap.constants import *
from mantrap.solver.objectives.acc_interact import InteractionAccelerationModule
from mantrap.solver.objectives.goal import GoalModule
from mantrap.solver.objectives.pos_interact import InteractionPositionModule

OBJECTIVES = {
    OBJECTIVE_INTERACTION_ACC: InteractionAccelerationModule,
    OBJECTIVE_GOAL: GoalModule,
    OBJECTIVE_INTERACTION_POS: InteractionPositionModule,
    OBJECTIVE_INTERACTION: InteractionPositionModule,  # default for interaction
}
