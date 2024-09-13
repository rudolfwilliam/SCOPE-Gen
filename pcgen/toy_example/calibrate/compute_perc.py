from pcgen.calibrate import compute_perc
from pcgen.predictors import compute_phat
from pcgen.distances import SquaredL2Distance
from pcgen.toy_example.paths import CKPT_DIR, DATA_DIR
from pcgen.nc_scores import CGScore
import torch


def main(epsilon, alpha):
    # distribution estimator
    distr_est = torch.load(CKPT_DIR + "/gmm.pt")
    # generated data to use for predictor estimation
    gen_data_pred = torch.load(DATA_DIR + "/data_gen_predest.pt")
    # generated data to use for conformal prediction
    gen_data_conf = torch.load(DATA_DIR + "/data_gen_calibration.pt")
    distance = SquaredL2Distance()
    phat = compute_phat(gen_data_pred, distr_est, distance, epsilon)
    perc = compute_perc(distr_est, phat, gen_data_conf, distance=distance, epsilon=epsilon, alpha=alpha)
    data_perc = {"count_percentile": perc, str(epsilon): epsilon, str(alpha): alpha}
    # save percentile
    torch.save(data_perc, DATA_DIR + "/pcgen_data/percentile.pt")
    nc_score = CGScore(phat)
    print("Predicted n: ", nc_score.largest_n_from_perc(perc))
    print("done.")

if __name__ == '__main__':
    main(epsilon=0.5, alpha=0.1)
    