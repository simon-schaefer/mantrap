from .acc_interact import InteractionAccelerationModule
from .goal import GoalModule
from .pos_interact import InteractionPositionModule

solver_module_dict = {
    "acc_interaction": InteractionAccelerationModule,
    "goal": GoalModule,
    "pos_interaction": InteractionPositionModule,
    "interaction": InteractionPositionModule,  # default for interaction
}
