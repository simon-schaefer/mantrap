import argparse
import importlib
import inspect
import logging
import os

from mantrap.utility.io import build_os_path


def main():
    parser = argparse.ArgumentParser(description="Test visualizations.")
    parser.add_argument("--function", type=str, default=None)
    args = parser.parse_args()

    test_directory = build_os_path("test/", make_dir=True)
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
        if args.function is not None and vis_name != args.function:
            logging.info(f"Skipping {vis_name}")
        else:
            logging.info(f"Running {vis_name} ...")
            vis_function()
            logging.info(f"Finishing up {vis_name}")


if __name__ == "__main__":
    main()
