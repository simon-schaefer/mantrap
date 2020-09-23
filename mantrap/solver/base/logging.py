import collections
import logging
import os
import re
import sys
import typing

import numpy as np
import pandas
import torch

import mantrap.constants


class OptimizationLogger:

    def __init__(self, is_logging: bool = False, is_debug: bool = False):
        """Logging variables. Using default-dict(deque) whenever a new entry is created, it does not have to be checked
        whether the related key is already existing, since if it is not existing, it is created with a queue as
        starting value, to which the new entry is appended. With an appending complexity O(1) instead of O(N) the
        deque is way more efficient than the list type for storing simple floating point numbers in a sequence.

        :param is_logging: whether logging is enable or not (default = False since slowing program down).
        """
        self._log = None
        self._iteration = None
        self._is_logging = is_logging
        self._is_debug = is_debug
        self.set_logging_preferences(is_logging=is_debug)

    def increment(self):
        """Increment internal iteration by 1 (`logger.iteration` is a property and therefore writing-protected,
        therefore `increment()` has to be used, so that the iteration can only be increased by 1 at a time.
        """
        self._iteration += 1

    @staticmethod
    def set_logging_preferences(is_logging: bool):
        log_level = logging.DEBUG if is_logging else logging.WARNING
        log_format = "[%(levelname)-6s > %(filename)-10s:%(lineno)4d (%(asctime)-8s:%(msecs)03d)] %(message)-s"
        formatter = logging.Formatter(log_format, datefmt="%H:%M:%S")

        logger = logging.getLogger()
        logger.setLevel(log_level)
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler())
        console = logger.handlers[0]
        console.setFormatter(formatter)
        console.setLevel(log_level)

        logging.getLogger("matplotlib").setLevel(logging.ERROR)
        logging.getLogger("numpy").setLevel(logging.WARNING)
        torch.set_printoptions(precision=4)
        np.set_printoptions(precision=4)
        np.set_printoptions(threshold=sys.maxsize)  # don't collapse arrays while printing
        np.seterr(divide='ignore', invalid='ignore')

    ###########################################################################
    # Writing log #############################################################
    ###########################################################################
    def log_reset(self):
        """Reset internal log, i.e. reset the internal iteration and logging dictionary."""
        self._iteration = 0
        if self.is_logging:
            self._log = collections.defaultdict(list)

    def log_update(self, kwarg_dict: typing.Dict[str, typing.Any]):
        """Append kwargs dictionary to internal log."""
        if self.is_logging:
            self._log.update(kwarg_dict)

    def log_append(self, tag: str = mantrap.constants.TAG_OPTIMIZATION, **kwargs):
        """Append to internal logging queue from the function's `**kwargs`.

        The values can either be none or a numpy/number/tensor-like object which is converted to a
        `torch.Tensor` in any case. The logging key is thereby a combination of the kwargs key and the
        given tag.

        :param tag: logging tag.
        :param kwargs: dictionary of elements to be added (key -> element).
        """
        if self.is_logging and self.log is not None:
            for key, value in kwargs.items():
                log_key = f"{tag}/{key}_{self._iteration:02d}"

                # If value is None, we assume (and check) that all previous values have been None
                # and append another None to this chain of None's.
                if value is None:
                    assert not any(self._log[key])
                    self._log[log_key].append(None)

                # Otherwise convert and concatenate the value to the stored torch tensor.
                else:
                    x = torch.tensor(value) if type(value) != torch.Tensor else value.detach()
                    x = x.unsqueeze(dim=0)  # make dim one larger for catting
                    if log_key not in self._log.keys():
                        self._log[log_key] = x
                    else:
                        self._log[log_key] = torch.cat((self._log[log_key], x), dim=0)

    ###########################################################################
    # Reading log #############################################################
    ###########################################################################
    def log_query(self, key: str = None, key_type: str = "", iteration: typing.Union[str, int] = "",
                  tag: str = None, apply_func: str = "cat",
                  ) -> typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor], None]:
        """Query internal log for some value with given key (log-key-structure: {tag}/{key_type}_{key}).

         :param key: query key, e.g. name of objective module.
         :param key_type: type of query (-> `mantrap.constants.LK_`...).
         :param iteration: optimization iteration to search in, if None then no iteration (summarized value).
         :param tag: logging tag to search in.
         :param apply_func: stack/concatenate tensors or concatenate last elements of results
                            if multiple results (shapes not checked !!) ["stack", "cat", "last", "as_dict"].
         """
        if not self.is_logging:
            raise LookupError("For querying the `is_logging` flag must be activate before solving !")
        assert self.log is not None
        if len(self.log.items()) == 0:
            raise LookupError("For querying the `solve()` method have to be called before !")
        if iteration == "end":
            iteration = str(self._iteration)
        if key is None:
            key = "(.*?)"
        if key_type is None:
            key_type = "(.*?)"

        # Search in log for elements that satisfy the query and return as dictionary. For the sake of
        # runtime the elements are stored as torch tensors in the log, therefore stack and list them.
        results_dict = {}
        query = f"{key_type}_{key}_{iteration}"
        if tag is not None:
            query = f"{tag}/{query}"
        for key, values in self.log.items():
            if re.search(query, key) is not None:
                results_dict[key] = values

        # Check whether all the resulting values are None, then return None.
        if all(x[0] is None for x in results_dict.values()):
            return None

        # If only one element is in the dictionary, return not the dictionary but the item itself.
        # Otherwise go through arguments one by one and apply them.
        if apply_func == "as_dict":
            return results_dict
        num_results = len(results_dict.keys())
        if num_results == 1:
            results = results_dict.popitem()[1]
        elif apply_func == "cat":
            results = torch.cat([results_dict[key] for key in sorted(results_dict.keys())])
        elif apply_func == "stack":
            results = torch.stack([results_dict[key] for key in sorted(results_dict.keys())], dim=0)
        elif apply_func == "last":
            results = torch.tensor([results_dict[key][-1] for key in sorted(results_dict.keys())])
        else:
            raise ValueError(f"Undefined apply function for log query {apply_func} !")
        return results.squeeze(dim=0)

    def log_store(self, csv_name: str = None) -> typing.Union[pandas.DataFrame, None]:
        """Store log for objective and constraints in CSV file (trajectories cannot be stored in DataFrame).

        :param csv_name: name of csv file = `csv_name.logging.csv`.
        """
        if self.log is None:
            return None
        log_types = [mantrap.constants.LT_OBJECTIVE, mantrap.constants.LT_CONSTRAINT]

        # Check whether there are any logged values  in internal log (e.g. some baselines do not
        # log objective/constraint values).
        log_check = self.log_query(key_type=mantrap.constants.LT_OBJECTIVE)
        if log_check is None:
            return None

        # Save the optimization performance for every optimization step into logging file. Since the
        # optimization log is `torch.Tensor` typed, it has to be mapped to a list of floating point numbers
        # first using the `map(dtype, list)` function.
        output_path = mantrap.constants.VISUALIZATION_DIRECTORY
        output_path = mantrap.utility.io.build_os_path(output_path, make_dir=True, free=False)
        output_path = os.path.join(output_path, f"{csv_name}.logging.csv")

        csv_log = {}
        for log_key_type in log_types:
            csv_log.update(self.log_query(key_type=log_key_type, apply_func="as_dict"))
        csv_log = {key: map(float, value) for key, value in csv_log.items()}

        df = pandas.DataFrame.from_dict(csv_log, orient='index')
        if csv_name is not None:
            df.to_csv(output_path)
        return df

    ###########################################################################
    # Logger properties #######################################################
    ###########################################################################
    @property
    def log(self) -> typing.Dict[str, typing.Union[torch.Tensor, typing.List[torch.Tensor]]]:
        return self._log

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def is_logging(self) -> bool:
        return self._is_logging

    @property
    def is_debug(self) -> bool:
        return self._is_debug
