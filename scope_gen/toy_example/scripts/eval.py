from itertools import product
from scope_gen.utils import load_config_from_json, set_seed
from scope_gen.toy_example.paths import CONFIG_DIR, CKPT_DIR, DATA_DIR
from scope_gen.algorithms.base import create_scorgen_pipeline, test
from scope_gen.toy_example.scripts.assess import assess_method
from scope_gen.nc_scores import SumScore
from scope_gen.toy_example.distances import L2Distance
import torch
import pickle


def main(cfg):
    parameter_combinations = list(product(cfg["data_set_sizes"], cfg["alpha_grid"]))
    # load generative model from disk
    gen_model = torch.load(CKPT_DIR + '/gmm.pt')
    distance_func = L2Distance()
    score_func = SumScore()
    # run experiments
    if VERBOSE:
        print(f"Running {len(parameter_combinations)} experiments.")
    for data_set_size, alpha in parameter_combinations:
        if VERBOSE:
            print(f"Running experiment with data set size {data_set_size} and alpha {alpha}.")
        coverages, sizes, first_adms = run_experiment(data_set_size, gen_model, epsilon=cfg["epsilon"], distance_func=distance_func, 
                                                      alpha=alpha, n_iterations=cfg["n_iterations"], n_coverage=cfg["n_coverage"],
                                                      score_func=score_func, split_ratios=[1/3]*3)
        # store results to disk using pickle
        if not DEBUG:
            with open(DATA_DIR + f'/eval_results_scorgen/results_{cfg["min_prob_threshold"]}_{data_set_size}_{alpha}.pkl', 'wb') as file:
                pickle.dump((coverages, sizes, first_adms), file)


def run_experiment(data_set_size, gen_model, epsilon, distance_func, n_iterations, n_coverage, split_ratios, alpha, score_func, k_max=20):
    coverages, sizes, first_adms = assess_method(n_iterations=n_iterations, n_coverage=n_coverage, gen_model=gen_model, method_func=create_scorgen_pipeline, test_func=test, 
                                                 data_set_size=data_set_size, epsilon=epsilon, distance_func=distance_func, k_max=k_max, split_ratios=split_ratios, 
                                                 alphas=[1 - (1 - alpha)**(1/3)]*3, first_adm_only=False, score_func=score_func, data_splitting=True, verbose=VERBOSE)
    if VERBOSE:
        print(f"Mean coverage: {sum(coverages)/n_iterations:.2f}")
    return coverages, sizes, first_adms


if __name__ == '__main__':
    set_seed(0)
    cfg = load_config_from_json(CONFIG_DIR + "/eval.json")
    global DEBUG, VERBOSE
    DEBUG = False 
    VERBOSE = True
    main(cfg)
