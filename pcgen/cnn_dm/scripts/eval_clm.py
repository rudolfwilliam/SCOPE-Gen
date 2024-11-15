from itertools import product
from pcgen.utils import load_config_from_json, set_seed
from pcgen.cnn_dm.paths import CONFIG_DIR, DATA_DIR
from pcgen.cnn_dm.paths import DATA_DIR
from pcgen.baselines.clm.uncertainty import run_experiment
import pickle
import numpy as np
import argparse
import os

USE_LAMBDA_1 = True # similarity
USE_LAMBDA_2 = True # quality
ALT_LAMBDA_1 = np.inf
ALT_LAMBDA_2 = -np.inf

DEBUG = False
VERBOSE = True

def eval(cfg, dir_, name="clm", score="count", reduced_max=20):
    data_dir = os.path.join(DATA_DIR, dir_)
    data_path = os.path.join(data_dir, "data.pkl")

    with open(data_path, 'rb') as file:
        data = pickle.load(file)

    parameter_combinations = list(product(cfg["data_set_sizes"], cfg["alpha_grid"]))
    
    # run experiments
    if VERBOSE:
        print(f"Running {len(parameter_combinations)} experiments.")
    for data_set_size, alpha in parameter_combinations:
        if VERBOSE:
            print(f"Running experiment with data set size {data_set_size} and alpha {alpha}.")

        experiment = {
            "data": data,
            "data_dir": DATA_DIR,
            "data_set_size": data_set_size,
            "n_iterations": cfg["n_iterations"],
            "n_coverage": cfg["n_coverage"],
            "alpha": alpha,
            "reduced_max": reduced_max,
            "score": score,
            "name": name,
            "verbose": VERBOSE,
            "debug": DEBUG,
            "use_lambda_1": USE_LAMBDA_1,
            "use_lambda_2": USE_LAMBDA_2,
            "alt_lambda_1": ALT_LAMBDA_1,
            "alt_lambda_2": ALT_LAMBDA_2
        }

        run_experiment(**experiment)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run the evaluation module with given parameters")
    parser.add_argument('--config', type=str, default=os.path.join("CONFIG_DIR", "eval.json"), help='Path to configuration JSON file')
    parser.add_argument('--name', type=str, default="CLM", help='name of the method')
    parser.add_argument('--dir', type=str, default="processed", help='Directory for processing')
    parser.add_argument('--score', type=str, default="count", help='Score type')
    parser.add_argument('--reduced_max', type=int, default=20, help='List of stages to process')

    args = parser.parse_args()

    set_seed(0)
    cfg = load_config_from_json(os.path.join(CONFIG_DIR, "eval.json"))

    eval(
         cfg, 
         dir_=args.dir, 
         name=args.name, 
         score=args.score, 
         reduced_max=args.reduced_max
         )
