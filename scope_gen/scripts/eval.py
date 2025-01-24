import os
import pickle
from itertools import product
import psutil

from scope_gen.algorithms.base import run_experiment
from scope_gen.baselines.clm.eval import eval_clm


def eval_all(cfg, cfgs_scope_gen, cfgs_clm, dir_, verbose=True, clm_only=False, scope_gen_only=False):
    if verbose:
        print(f"Running evaluation for {len(cfgs_scope_gen)} SCOPE-Gen configurations and {len(cfgs_clm)} CLM configurations.")
        print(f"Multiprocessing with {psutil.cpu_count(logical=False)} processes.")
    if not clm_only:
        for cfg_scope_gen in cfgs_scope_gen:
            eval(cfg=cfg, dir_=dir_, **cfg_scope_gen)
            if verbose:
                print(f"Finished evaluation for SCOPE-Gen configuration {cfg_scope_gen}.")
    else:
        print("Skipping SCOPE-Gen evaluation.")
    if not scope_gen_only:
        for cfg_clm in cfgs_clm:
            eval_clm(cfg, dir_, **cfg_clm)
            if verbose:
                print(f"Finished evaluation for CLM configuration {cfg_clm}.")
    else:
        print("Skipping CLM evaluation.")


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
    # evaluation of SCOPE-Gen
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
