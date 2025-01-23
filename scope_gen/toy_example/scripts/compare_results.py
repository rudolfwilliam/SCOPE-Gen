from scope_gen.toy_example.paths import DATA_DIR
import pickle
import matplotlib.pyplot as plt
import numpy as np


def main(min_prob_threshold, data_set_size, alpha):
    file_name = DATA_DIR + f'/eval_results_scope_gen/results_{min_prob_threshold}_{data_set_size}_{alpha}.pkl'
    with open(file_name, 'rb') as file:
        coverages, sizes, first_adms = pickle.load(file)
        coverages = np.array(coverages)
        sizes = np.array(sizes)
        coverages[coverages == 0.] = 1
        sizes[sizes == 0.] = float('inf')
    
    file_name = DATA_DIR + f'/eval_results_clm/results_{min_prob_threshold}_{data_set_size}_{alpha}.pkl'
    with open(file_name, 'rb') as file:
        coverages_clm, sizes_clm = pickle.load(file)
        coverages_clm = np.nan_to_num(np.array(coverages_clm), nan=1)
        sizes_clm = np.nan_to_num(np.array(sizes_clm), nan=float('inf'))

    # Replace infinite values with a large finite value for plotting
    max_finite_size = 8
    sizes[sizes == float('inf')] = max_finite_size + 1
    sizes_clm[sizes_clm == float('inf')] = max_finite_size + 1

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Histogram for coverages
    ax1.hist(coverages, bins=10, alpha=0.7, label='scope_gen')
    ax1.hist(coverages_clm, bins=10, alpha=0.7, label='CLM')
    ax1.set_xlabel('Mean Coverage')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    
    # Histogram for sizes with consistent binning
    bins = np.linspace(0, max_finite_size, 10).tolist() + [max_finite_size + 2]  # Add a bin for infinity
    
    ax2.hist(sizes, bins=bins, alpha=0.7, label='scope_gen')
    ax2.hist(sizes_clm, bins=bins, alpha=0.7, label='CLM')
    ax2.set_xlabel('Mean Size')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    # Adjust x-axis ticks to include a label for the "infinity" bin
    ax2.set_xticks(list(np.linspace(0, max_finite_size, 10)) + [max_finite_size + 1.5])
    ax2.set_xticklabels([f"{x:.1f}" for x in np.linspace(0, max_finite_size, 10)] + ['∞'])

    # Show the plots
    plt.tight_layout()
    plt.show()
    
    
if __name__ == '__main__':
    main(min_prob_threshold=0.0, data_set_size=300, alpha=0.1)
