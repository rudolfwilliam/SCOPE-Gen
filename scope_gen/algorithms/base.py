"""This script contains the main functions for the SCOPE-Gen algorithm."""

import time
import multiprocessing
import numpy as np
import psutil

from scope_gen.data.base import split_data_idxs
from scope_gen.calibrate.base import get_percentile
from scope_gen.order_funcs import quality_order_func, distance_order_func
from scope_gen.models.pipelines import (MinimalPredictionPipeline, 
                                        GenerationPredictionPipeline, 
                                        FilterPredictionPipeline, 
                                        RemoveDuplicatesPredictionPipeline)
from scope_gen.calibrate.score_computation import ScoreComputer
from scope_gen.nc_scores import (MaxScore, 
                                 MinScore, 
                                 DistanceScore, 
                                 CountScore, 
                                 SumScore)
from scope_gen.utils import store_results


SCORES_TABLE = {
                    "count": CountScore, 
                    "max": MaxScore,
                    "sum": SumScore
                }

SCORE_FUNC_QUALITY = MinScore()
SCORE_FUNC_DIST = DistanceScore()
ORDER_FUNC_QUALITY = quality_order_func
ORDER_FUNC_DIST = distance_order_func

# actually not required, but for the sake of consistency
MULTIPROCESSING = False


def create_scope_gen_pipeline(data, 
                              split_ratios, 
                              alphas, 
                              score, 
                              data_splitting=True, 
                              stages=None, 
                              count_adm=True, 
                              measure_time=False):
    """Create a SCOPE-Gen pipeline."""
    # Split the dataset into calibration for generation, calibration for quality pruning, and data for diversity pruning
    if stages is None:
        from scope_gen.data.meta_data import STAGES
        stages = STAGES
    if data_splitting:
        data_split_idxs = split_data_idxs(len(data), split_ratios)
    first_adms = []
    conformal_ps = []
    adjust_for_dupl = False
    if "remove_dupl" in stages:
        adjust_for_dupl = True
    if measure_time:
        start = time.time()
    score_func = SCORES_TABLE[score]()
    pipeline = MinimalPredictionPipeline(data)
    for i, stage in enumerate(stages):
        if not stage == "remove_dupl":
            conformal_p, first_adm = _calibrate_pipeline(pipeline, stage, stages, score_func, 
                                                         adjust_for_dupl, count_adm, data_split_idxs, alphas)
            if count_adm:
                first_adms.append(first_adm)
            conformal_ps.append(conformal_p)
        # Update the prediction pipeline
        pipeline = _update_pipeline(pipeline, stage, score_func, conformal_p)
    out = {"pipeline": pipeline, "conformal_ps": conformal_ps}

    if measure_time:
        end = time.time()
        out["time"] = end - start
    if count_adm:
        out["first_adms"] = np.stack(first_adms, axis=0).flatten()
    return out


def _calibrate_pipeline(pipeline, stage, stages, score_func, adjust_for_dupl, count_adm, data_split_idxs, alphas):
    score_computer = _initialize_score_computer(stage, pipeline, score_func, adjust_for_dupl)
    first_adm = None
    if count_adm:
        cal_scores, first_adm = score_computer.compute_scores(idxs=data_split_idxs[stages.index(stage)], return_idxs=True)
    else:
        cal_scores = score_computer.compute_scores(idxs=data_split_idxs[stages.index(stage)])
    # Compute the conformal quantile for generation
    conformal_p = get_percentile(cal_scores, alphas[stages.index(stage)])
    return conformal_p, first_adm


def _update_pipeline(pipeline, stage, score_func, conformal_p):
    if stage == "generation":
            pipeline = GenerationPredictionPipeline(pipeline, conformal_p, score_func)
    else:
        if stage == "quality":
            pipeline = FilterPredictionPipeline(pipeline, conformal_p, SCORE_FUNC_QUALITY, ORDER_FUNC_QUALITY)
        elif stage == "diversity":
            pipeline = FilterPredictionPipeline(pipeline, conformal_p, SCORE_FUNC_DIST, ORDER_FUNC_DIST)
        elif stage == "remove_dupl":
            pipeline = RemoveDuplicatesPredictionPipeline(pipeline)
    return pipeline


def _initialize_score_computer(stage, pipeline, score_func, adjust_for_dupl):
    if stage == "generation":
        score_computer = ScoreComputer(pipeline, score_func=score_func, order_func=None, 
                                        adjust_for_dupl=adjust_for_dupl)
    elif stage == "quality":
        score_computer = ScoreComputer(pipeline, score_func=SCORE_FUNC_QUALITY, 
                                        order_func=ORDER_FUNC_QUALITY, adjust_for_dupl=adjust_for_dupl)
    elif stage == "diversity":
        score_computer = ScoreComputer(pipeline, score_func=SCORE_FUNC_DIST, 
                                        order_func=ORDER_FUNC_DIST, adjust_for_dupl=adjust_for_dupl)
    else:
        raise ValueError(f"Stage {stage} not recognized.")

    return score_computer


def test(data, pipeline, return_std_coverages=False):
    """Test the pipeline."""
    pipeline.data = data
    coverages = []
    sizes = []
    for i in range(len(data)):
        prediction_set = pipeline.generate(i)
        if prediction_set is None:
            # "return everything" (reject)
            coverages.append(1)
            sizes.append(np.inf)
        else:
            coverages.append(int(any(prediction_set["labels"] == 1)))
            sizes.append(len(prediction_set["labels"]))
    if return_std_coverages:
        return np.mean(coverages), np.mean(sizes), np.std(coverages)  
    return np.mean(coverages), np.mean(sizes)


def experiment_iteration(args):
    data, data_set_size, n_coverage, split_ratios, alphas, score, stages, verbose, idx, return_std_coverages = args
    MEASURE_TIME = [False, True]
    COUNT_ADMS = [True, False]

    results = {
        "coverages": [],
        "sizes": [],
        "times": [],
        "first_adms": [],
        "std_coverages": []
    }

    idxs = np.random.permutation(len(data))
    cal_idxs = idxs[:data_set_size]
    test_idxs = idxs[data_set_size:(data_set_size + n_coverage)]
    data_cal = [data[idx] for idx in cal_idxs]

    for j in range(2):  # Two runs per iteration
        out = create_scope_gen_pipeline(data=data_cal, 
                                        split_ratios=split_ratios, 
                                        alphas=alphas,
                                        score=score, 
                                        data_splitting=True, 
                                        stages=stages, 
                                        count_adm=COUNT_ADMS[j], 
                                        measure_time=MEASURE_TIME[j])
        if return_std_coverages:
            coverage, size, std_coverage = test([data[idx] for idx in test_idxs], out["pipeline"], 
                                                return_std_coverages=return_std_coverages)
            results["std_coverages"].append(std_coverage)
        else:
            coverage, size = test([data[idx] for idx in test_idxs], out["pipeline"])

        if COUNT_ADMS[j]:
            results["first_adms"].append(np.mean(out["first_adms"]))
            results["coverages"].append(coverage)
            results["sizes"].append(size)
        if MEASURE_TIME[j]:
            results["times"].append(out["time"])
    
    return results


def run_experiment(data, 
                   data_dir, 
                   data_set_size, 
                   n_iterations, 
                   n_coverage, 
                   split_ratios, 
                   alpha, 
                   score, 
                   stages, 
                   name, 
                   verbose=False, 
                   debug=False, 
                   custom_path=None, 
                   alpha_params=None, 
                   return_std_coverages=False):
    
    """Fits the SCOPE-Gen pipeline to the data for many 
    iterations and one evaluation per fit."""
    
    K = len(split_ratios)

    if alpha_params is None:
        alpha_params = {"M" : 5, "parts" : [5 - (K - 1)] + [1 for _ in range(K-1)]}
    
    assert len(alpha_params["parts"]) == K
    
    alphas = compute_alphas(alpha, alpha_params)

    args = [(data, data_set_size, n_coverage, split_ratios, alphas, score, stages, verbose, i, return_std_coverages) for i in range(n_iterations)]
    
    num_processes = psutil.cpu_count(logical=False) if MULTIPROCESSING else 1
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(experiment_iteration, args)

    # Aggregate results from all iterations
    aggregate_results = {
        "coverages": [],
        "sizes": [],
        "times": [],
        "first_adms": [],
        "std_coverages": []
    }

    for result in results:
        aggregate_results["coverages"].extend(result["coverages"])
        aggregate_results["sizes"].extend(result["sizes"])
        aggregate_results["times"].extend(result["times"])
        aggregate_results["first_adms"].extend(result["first_adms"])
        if return_std_coverages:
            aggregate_results["std_coverages"].extend(result["std_coverages"])

    # Store results to disk unless debugging
    type = custom_path if custom_path else None
    if not debug:
        store_results(data_dir, name, alpha, score, aggregate_results["coverages"], aggregate_results["sizes"], 
                      aggregate_results["first_adms"], aggregate_results["times"], data_set_size, 
                      type=type, std_coverages=aggregate_results["std_coverages"])

    if verbose:
        print(f"Mean coverage: {np.mean(aggregate_results['coverages']):.2f}")


def compute_alphas(alpha, alpha_params):
    if len(alpha_params["parts"]) > 1:
        assert alpha_params["M"] == sum(alpha_params["parts"])
        chunk = (1 - alpha)**(1/alpha_params["M"])
        alphas = [(1 - chunk**alpha_params["parts"][j]) for j in range(len(alpha_params["parts"]))]
    else:
        alphas = [alpha]
    return alphas
