import collections
import os
import typing

import pandas
import torch

import mantrap.constants


class OptimizationLogger:

    def __init__(self, is_logging: bool = False):
        """Logging variables. Using default-dict(deque) whenever a new entry is created, it does not have to be checked
        whether the related key is already existing, since if it is not existing, it is created with a queue as
        starting value, to which the new entry is appended. With an appending complexity O(1) instead of O(N) the
        deque is way more efficient than the list type for storing simple floating point numbers in a sequence.

        :param is_logging: whether logging is enable or not (default = False since slowing program down).
        """
        self._log = None
        self._iteration = None
        self._is_logging = is_logging

    def increment(self):
        """Increment internal iteration by 1 (`logger.iteration` is a property and therefore writing-protected,
        therefore `increment()` has to be used, so that the iteration can only be increased by 1 at a time.
        """
        self._iteration += 1

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
                if value is None:
                    x = None
                else:
                    x = torch.tensor(value) if type(value) != torch.Tensor else value.detach()
                self._log[f"{tag}/{key}_{self._iteration}"].append(x)

    def log_summarize(self, csv_log_keys: typing.List[str], csv_name: str):
        """Summarize optimisation-step dictionaries to a single tensor per logging key, e.g. collapse all objective
        value tensors for k = 1, 2, ..., N to one tensor for the whole optimisation process.

        Attention: It is assumed that the last value of each series is the optimal value for this kth optimisation,
        e.g. the last value of the objective tensor `obj_overall` should be the smallest one. However it is hard to
        validate for the general logging key, therefore it is up to the user to implement it correctly.

        :param csv_log_keys: logging keys to write in output csv file.
        :param csv_name: name of csv file = `csv_name.logging.csv`.
        """
        if self.is_logging:
            assert self.log is not None

            # The log values have been added one by one during the optimization, so that they are lists
            # of tensors, stack them to a single tensor.
            for key, values in self.log.items():
                if type(values) == list and len(values) > 0 and all(type(x) == torch.Tensor for x in values):
                    self._log[key] = torch.stack(values)

            # Save the optimization performance for every optimization step into logging file. Since the
            # optimization log is `torch.Tensor` typed, it has to be mapped to a list of floating point numbers
            # first using the `map(dtype, list)` function.
            output_path = mantrap.constants.VISUALIZATION_DIRECTORY
            output_path = mantrap.utility.io.build_os_path(output_path, make_dir=True, free=False)
            output_path = os.path.join(output_path, f"{csv_name}.logging.csv")
            csv_log_k_keys = [f"{key}_{k}" for key in csv_log_keys for k in range(self._iteration + 1)]
            csv_log_k_keys += csv_log_keys

            csv_log = {key: map(float, self.log[key]) for key in csv_log_k_keys if key in self.log.keys()}
            pandas.DataFrame.from_dict(csv_log, orient='index').to_csv(output_path)

    ###########################################################################
    # Reading log #############################################################
    ###########################################################################
    def log_query(self, key: str, key_type: str, iteration: str = "", tag: str = None, apply_func: str = "cat",
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
        if iteration == "end":
            iteration = str(self._iteration)

        # Search in log for elements that satisfy the query and return as dictionary. For the sake of
        # runtime the elements are stored as torch tensors in the log, therefore stack and list them.
        results_dict = {}
        query = f"{key_type}_{key}_{iteration}"
        if tag is not None:
            query = f"{tag}/{query}"
        for key, values in self.log.items():
            if query not in key:
                continue
            results_dict[key] = values

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
