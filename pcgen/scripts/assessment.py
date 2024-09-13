"""This script collects all results from a dataset experiment and evaluates the final results."""

import os
import pickle
import numpy as np
from pcgen.scripts import DATA_DIRS

def main(score, data_set_size, alpha):
    aggr_results = {}
    for dataset_name, data_dir in DATA_DIRS.items():
        results = {}
        for dir in os.listdir(data_dir):
            method_name = dir
            dir_path = os.path.join(data_dir, dir)
            method_results = None
            for file in os.listdir(dir_path):
                file_name = file.split('.')[0].split('_')
                if score == file_name[-3] and str(data_set_size) == file_name[-2] and str(alpha).replace(".", "") == file_name[-1]:
                    file_path = os.path.join(dir_path, file)
                    with open(file_path, 'rb') as f:
                        method_results = pickle.load(f)
            if method_results is not None:
                results[method_name] = compute_mean_and_std(method_results)
        
        # Rearrange results to have method name before dataset name
        for method, metrics in results.items():
            if method not in aggr_results:
                aggr_results[method] = {}
            aggr_results[method][dataset_name] = metrics

    min_values = find_minimums(aggr_results)
    latex_table = generate_latex_table(aggr_results, min_values)
    print(latex_table)


def compute_mean_and_std(results):
    mean_adm_counts = sum(results['adm_counts']) / len(results['adm_counts'])
    no_rejects = (sum(np.isinf(results['sizes'])) + sum(np.isnan(results['sizes']))) / len(results['sizes'])
    sizes = [x for x in results['sizes'] if (not np.isinf(x) and not np.isnan(x))]
    mean_size = sum(sizes) / len(sizes)
    mean_time = sum(results['times']) / len(results['times'])
    
    std_adm_counts = (sum((x - mean_adm_counts) ** 2 for x in results['adm_counts']) / len(results['adm_counts'])) ** 0.5
    std_size = (sum((x - mean_size) ** 2 for x in sizes) / len(sizes)) ** 0.5
    std_time = (sum((x - mean_time) ** 2 for x in results['times']) / len(results['times'])) ** 0.5

    return {
        '\\# Queries': f"{mean_adm_counts:.3f} \pm {std_adm_counts:.3f}",
        'Time': f"{mean_time:.3f} \pm {std_time:.3f}",
        'Set Size': f"{mean_size:.3f} \pm {std_size:.3f}",
        '\\# Reject': f"{no_rejects:.3f} \pm 0.000"
    }

def find_minimums(aggr_results):
    min_values = {dataset: {metric: float('inf') for metric in ['\\# Queries', 'Time', 'Set Size', '\\# Reject']} for dataset in DATA_DIRS.keys()}
    for method_results in aggr_results.values():
        for dataset, dataset_results in method_results.items():
            for metric, value in dataset_results.items():
                if metric in min_values[dataset]:
                    mean_value = float(value.split(' ')[0])
                    if mean_value < min_values[dataset][metric]:
                        min_values[dataset][metric] = mean_value
    return min_values

def generate_latex_table(aggr_results, min_values):
    header = (
        "\\begin{table}[tb]\n"
        "\\centering\n"
        "\\caption{\\textbf{Quantitative Evaluations.} Three considered metrics $\\pm$ standard deviation of the different baselines from 100 repeated evaluations. For each evaluation, we first sub-sample a calibration set of size $n=300$ and calibrate each method at coverage level $\\alpha = 0.3$. Then, we assess each metric and take its means over a test set of size $500$, sampled from the remaining data. \\underline{Best is bold}.}\n"
        "\\label{tab:performance}\n"
        "\\resizebox{\\textwidth}{!}{\n"
        "\\begin{tabular}{ c || c || c | c | c | c }\n"
        "\\toprule\n"
        "\\textbf{Method} & \\textbf{Metric} & \\textbf{TriviaQA} & \\textbf{MIMIC-CXR} & \\textbf{CNN/DM} & \\textbf{Molecules} \\\\\n"
        "\\midrule\n"
    )
    content = ""

    # Assuming aggr_results is a dictionary like:
    # {'CLM': {'TriviaQA': {'Queries': '2.873 ± 0.349', 'Time': '0.006 ± 0.001', 'Set Size': '1.383 ± 0.284'}, ...}, ...}
    metrics = ['\\# Queries', 'Time', 'Set Size', '\\# Reject']
    for method, datasets in aggr_results.items():
        content += f"\\multirow{{4}}{{*}}{{{method}}}\n"
        for idx, metric in enumerate(metrics):
            content += " & "
            content += f"{metric} "
            for dataset in ['TriviaQA', 'MIMIC-CXR', 'CNN/DM', 'Molecules']:
                result = datasets.get(dataset, {}).get(metric, '')
                if result != '':
                    mean_value = float(result.split(' ')[0])
                    if mean_value == min_values[dataset][metric]:
                        result = f"\\bf{{{float(result.split(' ')[0]):.3f}}} \pm " + result.split(' ')[2]
                content += f"& {result} "
            content += "\\\\\n"
        content += "\\midrule\n" if method != list(aggr_results.keys())[-1] else "\\bottomrule\n"

    footer = (
        "\\end{tabular}\n"
        "}\n"
        "\\end{table}"
    )

    return header + content + footer

if __name__ == '__main__':
    main(score='max', data_set_size=600, alpha=0.3)
    