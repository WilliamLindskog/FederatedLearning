import argparse

from utils import run_assertions, get_data, get_args, get_config

# Get args
args = get_args()

# Get config
config = get_config()

# Get data
df = get_data(args.data_path)
print(df)
quit()