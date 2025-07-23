import sys
import argparse
import subprocess

from src.utils.config import get_cfg_from_file


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("hipt", add_help=add_help)
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    return parser


def classification(config_file):
    print("Running train/classification.py...")
    cmd = [
        sys.executable,
        "src/train/classification.py",
        "--config-file",
        config_file,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Classification failed. Exiting.")
        sys.exit(result.returncode)


def classification_multi(config_file):
    print("Running train/classification-multi.py...")
    cmd = [
        sys.executable,
        "src/train/classification-multi.py",
        "--config-file",
        config_file,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Multi-fold classification training failed. Exiting.")
        sys.exit(result.returncode)


def regression(config_file):
    print("Running train/regression.py...")
    cmd = [
        sys.executable,
        "src/train/regression.py",
        "--config-file",
        config_file,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Regression failed. Exiting.")
        sys.exit(result.returncode)


def regression_multi(config_file):
    print("Running train/regression-multi.py...")
    cmd = [
        sys.executable,
        "src/train/regression-multi.py",
        "--config-file",
        config_file,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Multi-fold regression training failed. Exiting.")
        sys.exit(result.returncode)


def survival(config_file):
    print("Running train/survival.py...")
    cmd = [
        sys.executable,
        "src/train/survival.py",
        "--config-file",
        config_file,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Survival training failed. Exiting.")
        sys.exit(result.returncode)


def survival_multi(config_file):
    print("Running train/survival-multi.py...")
    cmd = [
        sys.executable,
        "src/train/survival-multi.py",
        "--config-file",
        config_file,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Multi-fold survival training failed. Exiting.")
        sys.exit(result.returncode)


def main(args):

    config_file = args.config_file
    cfg = get_cfg_from_file(config_file)

    multi_fold = False
    if cfg.data.fold_dir is not None:
        multi_fold = True

    if cfg.task == "classification":
        if multi_fold:
            classification_multi(config_file)
        else:
            classification(config_file)
    elif cfg.task == "regression":
        if multi_fold:
            regression_multi(config_file)
        else:
            regression(config_file)
    elif cfg.task == "survival":
        if multi_fold:
            survival_multi(config_file)
        else:
            survival(config_file)
    else:
        print(f"Unsupported task: {cfg.task}. Exiting.")
        sys.exit(1)


if __name__ == "__main__":

    args = get_args_parser(add_help=True).parse_args()
    main(args)
