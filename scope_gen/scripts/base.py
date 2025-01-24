import os
import pickle
from itertools import product
from scope_gen.algorithms.base import run_experiment


def eval(cfg, 
         name, 
         score, 
         stages, 
         data_dir,
         dir_="processed",
         custom_path=None, 
         return_std_coverages=False,
         alpha_params=None,
         debug=False,
         verbose=True
         ):
    data_dir = os.path.join(data_dir, dir_)
    data_path = os.path.join(data_dir, "data.pkl")
    
    # duplicate removal does not need to be calibrated
    K = len(stages) - ('remove_dupl' in stages)
    
    with open(data_path, 'rb') as file:
        data = pickle.load(file)
    
    parameter_combinations = list(product(cfg["data_set_sizes"], cfg["alpha_grid"]))

    # run experiments
    if verbose:
        print(f"Running {len(parameter_combinations)} experiments.")
    for data_set_size, alpha in parameter_combinations:
        if verbose:
            print(f"Running experiment with data set size {data_set_size} and alpha {alpha}.")
        
        experiment_config = {
            "data": data,
            "data_dir": data_dir,
            "data_set_size": data_set_size,
            "n_iterations": cfg["n_iterations"],
            "split_ratios": [1/K] * K,
            "alpha": alpha,
            "score": score,
            "n_coverage": cfg["n_coverage"],
            "verbose": verbose,
            "debug": debug,
            "stages": stages,
            "name": name,
            "return_std_coverages": return_std_coverages,
            "custom_path": custom_path,
            "alpha_params": alpha_params
        }
        
        run_experiment(**experiment_config)
