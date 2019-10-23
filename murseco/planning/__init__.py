import numpy as np
import pyximport

pyximport.install(language_level=3, setup_args={"include_dirs": np.get_include()})

from murseco.planning.graph_search import time_expanded_graph_search
