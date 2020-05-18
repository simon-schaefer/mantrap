import typing

import matplotlib.pyplot as plt

from .atomics import __interactive_save_image


def visualize_curves(curve_dict: typing.Dict[str, typing.List[float]], y_label: str, file_path: str = None):
    """Plot and label several values over the iteration (list iteration), all together in one plot.
    Either save the image as png file (if `file_path` is set) or let the figure open for `plt.show()`.
    """
    assert len(curve_dict.keys()) > 0
    assert all([len(value_list) > 0 for value_list in curve_dict.values()])

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    for key, value_list in curve_dict.items():
        plt.plot(value_list, label=key)

    plt.grid()
    plt.legend()
    plt.xlabel("Iteration")
    plt.ylabel(y_label)
    return __interactive_save_image(fig, file_path=file_path)
