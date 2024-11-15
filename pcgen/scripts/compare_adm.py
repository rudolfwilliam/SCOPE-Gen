"""This scripts creates histograms that show how admissibility for CLM and SCAC-Gen compare, as sample size increases."""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
#import tikzplotlib
from pcgen.scripts import DATA_DIRS

SIZES = [300, 600, 1200]
SCORES = ['max', 'sum', 'count']
DATASET_NAME = "MIMIC-CXR"  # Specify your dataset name

def main(alpha):
    data_dir = DATA_DIRS[DATASET_NAME]
    alpha_str = str(alpha).replace(".", "")
    
    bin_edges = np.arange(0.55, 1.05, 0.03)  # Adjusted upper limit to include 1.0
    num_scores = len(SCORES)
    num_sizes = len(SIZES)
    
    # Create subplots
    fig, axes = plt.subplots(num_scores, num_sizes, figsize=(4 * num_sizes, 3 * num_scores), sharey=True)
    
    # Ensure axes is a 2D array for consistent indexing
    if num_scores == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_sizes == 1:
        axes = np.expand_dims(axes, axis=1)
    
    for score_idx, score in enumerate(SCORES):
        coverages_scacgen = []
        coverages_clm = []
        
        # Load data for each size
        for size in SIZES:
            # Load SCAC-Gen data
            file_name_scacgen = os.path.join(data_dir, "ourmethod{}", f'{score}_{size}_{alpha_str}.pkl')
            with open(file_name_scacgen, 'rb') as file:
                results = pickle.load(file)
                # Replace NaN values with 1.0
                coverage = np.nan_to_num(np.array(results['coverages']), nan=1.0)
                coverages_scacgen.append(coverage)
            # Load CLM data
            file_name_clm = os.path.join(data_dir, "CLM", f'{score}_{size}_{alpha_str}.pkl')
            with open(file_name_clm, 'rb') as file:
                results = pickle.load(file)
                # Replace NaN values with 1.0
                coverage = np.nan_to_num(np.array(results['coverages']), nan=1.0)
                coverages_clm.append(coverage)
        
        # Plot histograms for each size
        for size_idx, size in enumerate(SIZES):
            ax = axes[score_idx, size_idx]
            # Histogram for coverages
            n_scacgen, bins_scacgen, patches_scacgen = ax.hist(
                coverages_scacgen[size_idx], bins=bin_edges, alpha=0.7, density=True,
                label='SCAC-Gen'
            )
            n_clm, bins_clm, patches_clm = ax.hist(
                coverages_clm[size_idx], bins=bin_edges, alpha=0.7, density=True,
                label='CLM'
            )
            ax.axvline(1 - alpha, color='red', linestyle='--', linewidth=2, label='$1 - \\alpha$')
            
            # Calculate means
            mean_scacgen = np.mean(coverages_scacgen[size_idx])
            mean_clm = np.mean(coverages_clm[size_idx])
            
            # Plot vertical dashed lines for means
            ax.axvline(mean_scacgen, linestyle='dashed', linewidth=2, label='SCAC-Gen Mean')
            ax.axvline(mean_clm, color="orange", linestyle='dashed', linewidth=2, label='CLM Mean')
            
            # Set labels and titles
            ax.set_ylabel('Frequency')
            if size_idx == num_sizes // 2:
                ax.set_title(f'Non-conformity measure: {score}', fontsize=12)
            if score_idx == num_scores - 1:
                ax.set_xlabel('Mean Admissibility')
            # Adjust legend to only display unique entries
            if score_idx == 0 and size_idx == num_sizes - 1:
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), fontsize='small', loc='upper right')
    
    plt.tight_layout()
    plt.show()
    # Uncomment the following lines if you need to save the figure with TikZ
    #from pcgen.utils import tikzplotlib_fix_ncols
    #tikzplotlib_fix_ncols(fig)
    #tikzplotlib.save("coverage_comparison.tex")
    

if __name__ == '__main__':
    main(alpha=0.3)
