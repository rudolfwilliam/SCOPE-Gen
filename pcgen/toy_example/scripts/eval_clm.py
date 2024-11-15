from itertools import product
from pcgen.utils import load_config_from_json, set_seed
from pcgen.toy_example.paths import CONFIG_DIR, CKPT_DIR, DATA_DIR
from pcgen.toy_example.distances import L2Distance
from pcgen.baselines.clm.uncertainty import create_clm_pipeline, test
from pcgen.utils import create_parameter_grid_clm
from pcgen.toy_example.scripts.assess import assess_method
import torch
import pickle
import numpy as np


def main(cfg, verbose=True):
    parameter_combinations = list(product(cfg["data_set_sizes"], cfg["alpha_grid"]))
    # load generative model from disk
    gen_model = torch.load(CKPT_DIR + '/gmm.pt')
    #score_func = SumScore()
    distance_func = L2Distance()
    #admission_func = ProximalAdmission(epsilon=cfg["epsilon"], distance_func=distance_func)
    # run experiments
    if verbose:
        print(f"Running {len(parameter_combinations)} experiments.")
    for data_set_size, alpha in parameter_combinations:
        if verbose:
            print(f"Running experiment with data set size {data_set_size} and alpha {alpha}.")
        coverages, sizes = run_experiment(data_set_size, alpha, gen_model, epsilon=cfg["epsilon"], distance_func=distance_func, 
                                          n_iterations=cfg["n_iterations"], n_coverage=cfg["n_coverage"], k_max=cfg["k_max"])
        # store results to disk using pickle
        with open(DATA_DIR + f'/eval_results_clm/results_{cfg["min_prob_threshold"]}_{data_set_size}_{alpha}.pkl', 'wb') as file:
            pickle.dump((coverages, sizes), file)


def run_experiment(data_set_size, alpha, gen_model, epsilon, distance_func, n_iterations, n_coverage, k_max=20):
    if OPTIM:
        # optimize over parameter grid
        delta_1, delta_2 = create_parameter_grid_clm(level=alpha, n_points=10, min_value=alpha/15, max_value=alpha/5)
        min_size = np.inf
        best_config = None
        if VERBOSE:
            print(f"Optimizing over delta_1 and delta_2.")
        for i in range(len(delta_1)):
            coverages = []
            sizes = []
            coverages, sizes = assess_method(n_iterations=n_iterations, n_coverage=n_coverage, gen_model=gen_model, 
                                            method_func=create_clm_pipeline, test_func=test, data_set_size=data_set_size, 
                                            epsilon=epsilon, distance_func=distance_func, k_max=k_max, split_ratio=0.5, 
                                            first_adm_only=FIRST_ADM_ONLY, delta_1=delta_1[i], delta_2=delta_2[i])
            sizes = np.array(sizes)
            sizes_wo_nan = sizes[~np.isnan(sizes)]
            mean_size = np.mean(sizes_wo_nan)
            if mean_size < min_size:
                min_size = mean_size
                best_config = (delta_1[i], delta_2[i])
                best_coverages = coverages
                best_sizes = sizes
    else:
        # run with fixed parameters
        best_config = (alpha - alpha/10, alpha/10)
        best_coverages, best_sizes = assess_method(n_iterations=n_iterations, n_coverage=n_coverage, gen_model=gen_model, 
                                                   method_func=create_clm_pipeline, test_func=test, data_set_size=data_set_size, 
                                                   epsilon=epsilon, distance_func=distance_func, k_max=k_max, split_ratio=0.5, 
                                                   first_adm_only=FIRST_ADM_ONLY, delta_1=alpha, delta_2=alpha)
    if VERBOSE:
        print(f"Best configuration - delta_1: {best_config[0]:.2f}, delta_2: {best_config[1]:.2f}.")
        print(f"Mean coverage: {sum(best_coverages)/n_iterations:.2f}")

    return best_coverages, best_sizes


if __name__ == '__main__':
    set_seed(0)
    cfg = load_config_from_json(CONFIG_DIR + "/eval.json")
    global DEBUG, VERBOSE, FIRST_ADM_ONLY, OPTIM
    # prevents the script from saving results to disk
    DEBUG = False
    VERBOSE = True
    # this variable must be set to True for biased CLM
    FIRST_ADM_ONLY = False
    # whether to optimize over delta_1 and delta_2 or choose fixed values
    OPTIM = True
    main(cfg)
