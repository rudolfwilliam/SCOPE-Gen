from itertools import product
import pickle
import os

from scope_gen.baselines.clm.uncertainty import run_experiment


def eval(cfg, 
         dir_, 
         use_lamdba_1,
         use_lambda_2,
         alt_lambda_1,
         alt_lambda_2,
         data_dir,
         name="clm", 
         score="count", 
         reduced_max=20,
         verbose=True,
         debug=False
         ):
    data_dir = os.path.join(data_dir, dir_)
    data_path = os.path.join(data_dir, "data.pkl")

    with open(data_path, 'rb') as file:
        data = pickle.load(file)

    parameter_combinations = list(product(cfg["data_set_sizes"], cfg["alpha_grid"]))
    
    # run experiments
    if verbose:
        print(f"Running {len(parameter_combinations)} experiments.")
    for data_set_size, alpha in parameter_combinations:
        if verbose:
            print(f"Running experiment with data set size {data_set_size} and alpha {alpha}.")

        experiment = {
            "data": data,
            "data_dir": data_dir,
            "data_set_size": data_set_size,
            "n_iterations": cfg["n_iterations"],
            "n_coverage": cfg["n_coverage"],
            "alpha": alpha,
            "reduced_max": reduced_max,
            "score": score,
            "name": name,
            "verbose": verbose,
            "debug": debug,
            "use_lambda_1": use_lamdba_1,
            "use_lambda_2": use_lambda_2,
            "alt_lambda_1": alt_lambda_1,
            "alt_lambda_2": alt_lambda_2
        }

        run_experiment(**experiment)
