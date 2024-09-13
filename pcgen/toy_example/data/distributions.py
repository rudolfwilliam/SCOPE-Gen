"""Known distribution for toy example (Bart Simpson distribution)."""

import torch
from scipy.interpolate import interp1d
from torch.distributions import Normal
from pcgen.toy_example.models.gaussian_mixture import GaussianMixtureModel
from pcgen.data import Distribution


class ThreeGaussians(GaussianMixtureModel):
    """Three Gaussians distribution."""
    def __init__(self, num_components=3, num_features=2):
        super(ThreeGaussians, self).__init__(num_components=num_components, num_features=num_features)
        self.means = torch.tensor([[-2., -2.], [-2., 2.], [1., 1.]])
        self.covs = torch.tensor([[[1., 0.], [0., 1.]], [[2., 0.], [0., 0.5]], [[1., 0.], [0., 1.]]])
        self.weights = torch.tensor([1/3, 1/4, 1/2])


NEG_LIM = -5.0
POS_LIM = 5.0
DX = 0.01


class BartSimpsonDistribution(Distribution):
    """Bart Simpson distribution."""
    def __init__(self):
        super(BartSimpsonDistribution, self).__init__()
        self.inv_cdf = self.compute_inv_cdf()
    
    def prob(self, x):
        prob = (1/2) * Normal(0, 1).log_prob(x).exp()
        for i in range(0, 5):
            prob += (1/10) * Normal((i/2) - 1, 1/10).log_prob(x).exp()
        return prob
    
    def sample(self, num_samples):
        """Uniform samples from the distribution."""
        samples = torch.rand((num_samples,))
        # obtain samples from the inverse cdf
        return self.inv_cdf(samples).unsqueeze(1)
    
    def compute_inv_cdf(self):
        """Cumulative distribution function."""
        rg = torch.arange(NEG_LIM, POS_LIM, DX)
        cdf = torch.zeros(rg.size())
        for i in range(rg.size(0)):
            cdf[i] = self.prob(rg[i]) + cdf[i-1]
        cdf = cdf / cdf[-1]
        inv_cdf = invert_function(cdf, rg)
        return inv_cdf
        

def invert_function(func_tensor, rg):
    # Define the interpolation function
    interp_func = interp1d(func_tensor.numpy(), rg.numpy(), kind='linear')

    # Define the inverse function
    def inverse_func(y):
        # Ensure y is within the range of the function
        y_clipped = torch.clamp(y, func_tensor[0], func_tensor[-1])
        # Interpolate to find the corresponding x value
        return torch.tensor(interp_func(y_clipped.cpu().numpy()))

    return inverse_func
    