from mantrap_evaluation.datasets.custom.haruki import scenario_custom_haruki
from mantrap_evaluation.datasets.custom.parallel import scenario_custom_parallel
from mantrap_evaluation.datasets.eth import scenario_eth


SCENARIOS = {
    "custom_haruki": scenario_custom_haruki,
    "custom_parallel": scenario_custom_parallel,
    "eth": scenario_eth,
}
