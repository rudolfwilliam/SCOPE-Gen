import os
import pickle
from pcgen.scripts import SINGLE_RUN_DIRS

EXCLUDED_METHODS = []
DATA_SET_SIZES = [300, 600, 1200]
ALPHAS = [0.3, 0.35, 0.4]

def main(score, experiment):
    dir = SINGLE_RUN_DIRS[experiment]
    for alpha in ALPHAS:
        for data_set_size in DATA_SET_SIZES:
            # loop through directory
            for folder in os.listdir(dir):
                method_name = folder
                if method_name in EXCLUDED_METHODS:
                    continue
                for file in os.listdir(os.path.join(dir, folder)):
                    file_name = file.split('.')[0].split('_')
                    if score == file_name[-3] and str(data_set_size) == file_name[-2] and str(alpha).replace(".", "") == file_name[-1]:
                        file_path = os.path.join(dir, method_name, file)
                        with open(file_path, 'rb') as f:
                            res = pickle.load(f)
                            mean, std = compute_mean_and_std_cov(res)
                            print(f"method: {method_name}, dataset: {data_set_size}, alpha: {alpha}")
                            # round to 2 decimal places
                            print(f"{mean:.2f} \pm {std:.2f}")

def compute_mean_and_std_cov(res):
    # compute mean and std of cov
    assert len(res['std_coverages']) <= 2
    mean_cov = res['coverages'][0].item()
    assert res['std_coverages'] is not None
    std_cov = res['std_coverages'][0]
    return mean_cov, std_cov

if __name__ == '__main__':
    main(score='sum', experiment="Molecules")
    