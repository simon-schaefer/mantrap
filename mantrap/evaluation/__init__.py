from typing import Any, Dict

import importlib
import inspect


# Aggregate metrics automatically as dictionary mapping the metric names to its function call,
# using the importlib and inspect module. In order to differentiate the metric functions from other, maybe imported,
# functions, they have to uniquely start with "metric_", e.g. a distance metric should be called "metric_distance".
def evaluate_metrics(**metric_kwargs) -> Dict[str, Any]:
    metrics = {}
    module = importlib.__import__("metrics")
    functions = [o for o in inspect.getmembers(module) if inspect.isfunction(o[1])]
    for function_tuple in functions:
        function_name, _ = function_tuple
        if function_name.startswith("metric_"):
            function_tag = function_name.replace("metric_", "")
            metrics[function_tag] = function_tuple[1]

    return {tag: function(**metric_kwargs) for tag, function in metrics.items()}
