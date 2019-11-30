import importlib
import inspect
import logging
import os

from mantrap.utility.io import path_from_home_directory


def main():
    test_directory = path_from_home_directory("test/")
    test_files = [path for path in os.listdir(test_directory) if path.endswith(".py") and path.startswith("test_")]

    vis_functions = {}
    for test_file in test_files:
        module = importlib.__import__(test_file.replace(".py", ""))
        functions = [o for o in inspect.getmembers(module) if inspect.isfunction(o[1])]
        for function_tuple in functions:
            function_name, _ = function_tuple
            if function_name.startswith("visualize_"):
                vis_functions[function_name] = function_tuple[1]

    for vis_name, vis_function in vis_functions.items():
        logging.info(f"Running {vis_name} ...")
        vis_function()
        logging.info(f"Finishing up {vis_name}")


if __name__ == "__main__":
    main()
