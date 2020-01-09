import numpy as np

import mantrap.utility.datasets as datasets
from mantrap.utility.shaping import check_ado_trajectories


def test_eth():
    trajectories, ado_id_dict = datasets.load_eth(return_id_dict=True)
    assert check_ado_trajectories(trajectories, num_modes=1)
    assert np.isclose(trajectories[ado_id_dict[1], 0, 0, 0], 11.238836854)
    assert np.isclose(trajectories[ado_id_dict[1], 0, 0, 1], 3.7469588555)
    assert np.isclose(trajectories[ado_id_dict[401], 0, 0, 0], 7.32418581497)
    assert np.isclose(trajectories[ado_id_dict[401], 0, 0, 1], 11.5750002861)
    assert np.isclose(trajectories[ado_id_dict[2], 0, 2, 0], 10.4891601335)
    assert np.isclose(trajectories[ado_id_dict[2], 0, 2, 1], 3.08730316909)
