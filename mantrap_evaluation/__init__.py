import typing

import mantrap


# Aggregate all metric names from metrics module.
def get_metrics() -> typing.Dict[str, typing.Callable]:
    return mantrap.utility.io.load_functions_from_module(module="mantrap_evaluation.metrics", prefix="metric_")


# Aggregate metrics automatically as dictionary mapping the metric names to its function call,
# using the importlib and inspect module. In order to differentiate the metric functions from other, maybe imported,
# functions, they have to uniquely start with "metric_", e.g. a distance metric should be called "metric_distance".
def evaluate_metrics(**metric_kwargs) -> typing.Dict[str, float]:
    metrics = get_metrics()
    assert len(metrics.keys()) > 0
    return {tag: function(**metric_kwargs) for tag, function in metrics.items()}
