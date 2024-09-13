"""Fit a GMM to the data using the EM algorithm."""

from pcgen.toy_example.paths import DATA_DIR, CKPT_DIR
from pcgen.toy_example.models import GaussianMixtureModel as GaussianMixture_torch
from sklearn.mixture import GaussianMixture as GaussianMixture_np
import torch

def main():
    data = torch.load(DATA_DIR + '/data_gen_model.pt')
    gmm = GaussianMixture_np(n_components=5)
    gmm.fit(data)
    # convert to torch model
    gmm_torch = GaussianMixture_torch(num_components=5)
    gmm_torch.means = torch.nn.Parameter(torch.tensor(gmm.means_).float())
    gmm_torch.covs = torch.nn.Parameter(torch.tensor(gmm.covariances_).float())
    gmm_torch.weights = torch.nn.Parameter(torch.tensor(gmm.weights_).float())
    # save model as a pt file
    torch.save(gmm_torch, CKPT_DIR + '/gmm.pt')
    print("done.")

if __name__ == '__main__':
    main()
