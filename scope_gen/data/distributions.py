
class Distribution(object):
    
    def prob(self, x):
        """Probability density."""
        raise NotImplementedError
    
    def sample(self, num_samples):
        """Sample from the distribution."""
        raise NotImplementedError
    