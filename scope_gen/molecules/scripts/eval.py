from itertools import product
from scope_gen.utils import load_config_from_json, set_seed
from scope_gen.molecules.paths import CONFIG_DIR, DATA_DIR
from scope_gen.algorithms.base import run_experiment
from scope_gen.molecules.paths import DATA_DIR
import argparse
import pickle
import os

DEBUG = False
VERBOSE = True

def eval(cfg, name, score, stages, dir_="processed", custom_path=None, return_std_coverages=False):
    data_dir = os.path.join(DATA_DIR, dir_)
    data_path = os.path.join(data_dir, "data.pkl")
    
    # duplicate removal does not need to be calibrated
    K = len(stages) - ('remove_dupl' in stages)
    
    with open(data_path, 'rb') as file:
        data = pickle.load(file)
    
    parameter_combinations = list(product(cfg["data_set_sizes"], cfg["alpha_grid"]))

    # run experiments
    if VERBOSE:
        print(f"Running {len(parameter_combinations)} experiments.")
    for data_set_size, alpha in parameter_combinations:
        if VERBOSE:
            print(f"Running experiment with data set size {data_set_size} and alpha {alpha}.")
        
        experiment_config = {
            "data": data,
            "data_dir": DATA_DIR,
            "data_set_size": data_set_size,
            "n_iterations": cfg["n_iterations"],
            "split_ratios": [1/K] * K,
            "alpha": alpha,
            "score": score,
            "n_coverage": cfg["n_coverage"],
            "verbose": VERBOSE,
            "debug": DEBUG,
            "stages": stages,
            "name": name,
            "custom_path": custom_path,
            "return_std_coverages": return_std_coverages
        }
        
        run_experiment(**experiment_config)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run the evaluation module with given parameters")
    parser.add_argument('--config', type=str, default=os.path.join("CONFIG_DIR", "eval.json"), help='Path to configuration JSON file')
    parser.add_argument('--name', type=str, default="SCOPE-Gen", help='name of the method')
    parser.add_argument('--dir', type=str, default="processed", help='Directory for processing')
    parser.add_argument('--score', type=str, default="count", help='Score type')
    parser.add_argument('--custom_path', type=str, default=None, help='Custom path to storing the result')
    parser.add_argument('--return_std_coverages', type=bool, default=False, help='Return standard deviations of coverages')
    parser.add_argument('--stages', nargs='+', default=["generation", "quality", "remove_dupl"], help='List of stages to process')
    args = parser.parse_args()
    
    set_seed(0)
    cfg = load_config_from_json(args.config)  # Use args.config instead of hardcoded path
    eval(
        cfg,
        dir_=args.dir,
        name=args.name,
        score=args.score,
        stages=args.stages,
        return_std_coverages=args.return_std_coverages,
        custom_path=args.custom_path
    )
