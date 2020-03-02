import argparse
import importlib
import inspect
import logging
import os

from mantrap.utility.io import build_os_path, load_functions_from_module


def main():
    parser = argparse.ArgumentParser(description="Test visualizations.")
    parser.add_argument("--function", type=str, default=None)
    args = parser.parse_args()

    vis_functions = {}
    test_directory = build_os_path("test/", make_dir=True)
    test_files = [path for path in os.listdir(test_directory) if path.endswith(".py") and path.startswith("test_")]
    for test_file in test_files:
        vis_functions.update(load_functions_from_module(module=test_file.replace(".py", ""), prefix="visualize_"))

    for vis_name, vis_function in vis_functions.items():
        if args.function is not None and vis_name != args.function:
            logging.info(f"Skipping {vis_name}")
        else:
            logging.info(f"Running {vis_name} ...")
            vis_function()
            logging.info(f"Finishing up {vis_name}")


if __name__ == "__main__":
    main()
