"""Simple Gaussian Mixture Model as density estimator."""

import torch
import numpy as np
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal


class GaussianMixtureModel(nn.Module):
    def __init__(self, num_components=6, num_features=2):
        super(GaussianMixtureModel, self).__init__()
        self.num_components = num_components
        self.num_features = num_features
        self.means = torch.randn(num_components, num_features)
        self.covs = torch.randn(num_components, num_features, num_features)
        self.weights = torch.randn(num_components)
    
    def forward(self, x):
        return self.prob(x).numpy()
    
    def prob(self, x):
        x = torch.tensor(x)
        prob = torch.zeros(x.shape[0])
        for i in range(self.num_components):
            prob += self.weights[i] * MultivariateNormal(self.means[i], self.covs[i]).log_prob(x).exp()
        return prob.detach().numpy()
    
    def predictor(self, x, cond=None):
        # if x is a single point, convert it to a batch
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)
        return self.prob(x)
    
    def sample(self, num_samples, cond=None, min_prob_threshold=0.0, max_resample_attempts=10):
        samples = []
        attempts = 0

        while len(samples) < num_samples and attempts < max_resample_attempts * num_samples:
            # Sample indices according to the weights
            idx = torch.multinomial(self.weights, num_samples - len(samples), replacement=True)
            
            # Gather the means and covariances for the sampled indices
            selected_means = self.means[idx]
            selected_covs = self.covs[idx]
            
            # Create a batch of MultivariateNormal distributions
            dists = MultivariateNormal(selected_means, selected_covs)
            
            # Sample a batch of points
            new_samples = dists.sample()
            
            # Calculate the probability density for each sample
            probabilities = torch.exp(dists.log_prob(new_samples))
            
            # Filter out samples with probabilities lower than the threshold
            valid_samples = new_samples[probabilities >= min_prob_threshold]
            
            samples.extend(valid_samples.numpy())
            attempts += 1

        if len(samples) < num_samples:
            print(f"Warning: Only {len(samples)} valid samples were generated after {attempts} attempts.")

        return np.array(samples[:num_samples])
        