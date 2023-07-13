import yaml
import pandas as pd
import argparse

def run_assertions(config : dict = None) -> None:
    """ Run assertions for main script. """
    if config is not None:
        assert config['task'] in ['regression', 'classification']
        print("Done")

def get_config() -> dict:
    """ Get config for main script. """
    with open("./etc/config.yaml", "r") as f:
        config = yaml.safe_load(f)
        run_assertions(config)
    return config

def get_args() -> argparse.Namespace:
    """ Get arguments for main script. """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default="./dataset/test.csv",
        help="Path to the dataset."
    )
    args = parser.parse_args()
    return args

def get_data(data_path : str) -> pd.DataFrame:
    """ Get data from path. """
    data = pd.read_csv(data_path)
    return data