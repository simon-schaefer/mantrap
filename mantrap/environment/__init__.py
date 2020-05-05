from mantrap.environment.base.graph_based import GraphBasedEnvironment

from mantrap.environment.social_forces import SocialForcesEnvironment
from mantrap.environment.trajectron import Trajectron
from mantrap.environment.simplified.kalman import KalmanEnvironment
from mantrap.environment.simplified.orca import ORCAEnvironment
from mantrap.environment.simplified.potential_field import PotentialFieldEnvironment

ENVIRONMENTS = [
    SocialForcesEnvironment,
    Trajectron,
    KalmanEnvironment,
    ORCAEnvironment,
    PotentialFieldEnvironment
]
ENVIRONMENTS_DICT = {env.environment_name(): env for env in ENVIRONMENTS}
