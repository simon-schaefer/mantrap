import typing

import matplotlib.pyplot as plt
import numpy as np

from .atomics import __interactive_save_image


def visualize_curves(
    curve_dict: typing.Dict[str, typing.List[float]],
    y_label: str,
    scale: str = "linear",
    file_path: str = None
):
    """Plot and label several values over the iteration (list iteration), all together in one plot.
    Either save the image as png file (if `file_path` is set) or let the figure open for `plt.show()`.

    :param curve_dict: dictionary of values to plot which keys as labels.
    :param y_label: y-axis label.
    :param scale: value scale function ("linear", "log").
    :param file_path: image storage path, if None then figure will be let opened to `plt.show()`.
    """
    scale_func_dict = {"linear": lambda x: x,
                       "log": np.log
                       }  # type: typing.Dict[str, typing.Callable[[typing.List[float]], typing.List[float]]]

    assert len(curve_dict.keys()) > 0
    assert all([len(value_list) > 0 for value_list in curve_dict.values()])
    assert scale in scale_func_dict.keys()

    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    for key, value_list in curve_dict.items():
        plt.plot(scale_func_dict[scale](value_list), label=key)

    plt.grid()
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel(f"{y_label} ({scale})")
    return __interactive_save_image(fig, file_path=file_path)
