"""This scripts creates histograms that show how coverage for CLM and SCAC-Gen compare, as sample size increases."""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tikzplotlib
from pcgen.scripts import DATA_DIRS

SIZES = [300, 600, 1200]

def main(dataset_name, score, alpha):
    data_dir = DATA_DIRS[dataset_name]
    alpha_str = str(alpha).replace(".", "")
    
    coverages_scacgen = []

    for size in SIZES:
        file_name = os.path.join(data_dir, "ourmethod{}", f'{score}_{size}_{alpha_str}.pkl')
        with open(file_name, 'rb') as file:
            results = pickle.load(file)
            #coverage = np.nan_to_num(np.array(results['coverages']), nan=1)
            coverage = np.array(results['coverages'])[~np.isnan(results['coverages'])]
            coverages_scacgen.append(coverage)

        #sizes_scacgen = np.nan_to_num(np.array(results['sizes']), nan=float('inf'))
    
    coverages_clm = []

    for size in SIZES:
        file_name = os.path.join(data_dir, "CLM", f'{score}_{size}_{alpha_str}.pkl')
        with open(file_name, 'rb') as file:
            results = pickle.load(file)
            #coverage = np.nan_to_num(np.array(results['coverages']), nan=1)
            coverage = np.array(results['coverages'])[~np.isnan(results['coverages'])]
            coverages_clm.append(coverage)

    # Create subplots
    fig, axes = plt.subplots(1, len(SIZES), figsize=(12, 6), sharey=True)
    bin_edges = np.arange(0.55, 1, 0.03)

    for i, size in enumerate(SIZES):
        # Histogram for coverages
        axes[i].hist(coverages_scacgen[i], bins=bin_edges, alpha=0.7, density=True, label='\\ourmethod{}')
        axes[i].hist(coverages_clm[i], bins=bin_edges, alpha=0.7, density=True, label='CLM')
        axes[i].set_xlabel('Mean Coverage')
        if i == 0:
            axes[i].set_ylabel('Frequency')
        axes[i].axvline(1 - alpha, color='r', linestyle='--', linewidth=2, label='$1 - \\alpha$')
        axes[i].legend()

    # Show the plots
    #plt.tight_layout()
    plt.show()
    #from pcgen.utils import tikzplotlib_fix_ncols
    #tikzplotlib_fix_ncols(fig)
    #tikzplotlib.save("coverage_comparison.tex")


if __name__ == '__main__':
    main(dataset_name="CNN/DM", score="max", alpha=0.3)
