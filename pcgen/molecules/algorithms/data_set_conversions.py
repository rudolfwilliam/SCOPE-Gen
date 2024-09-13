from pcgen.calibrate.score_computation import CGDataConverter
from pcgen.molecular_extensions.data.conversions import clean_and_convert_samples
from pcgen.utils import override


class MoleculeCGDataConverter(CGDataConverter):
    """We want to do sampling in batches rather than incremenetally for computational efficiency."""
    def __init__(self, gen_model, distance_func, epsilon, batch_size=256):
        self.batch_size = batch_size
        super().__init__(gen_model, distance_func, epsilon, verbose=True)
    
    @override
    def _convert_instance(self, instance):
        """Sample entire batch (maximum prediction set size) and check for the first sample in epsilon radius.
           If not found, simply return inf."""
        z_gt = instance[0]
        cond = instance[1]
        samples = self.gen_model.sample(self.batch_size, cond)
        # remove invalid samples and duplicates
        samples = clean_and_convert_samples(samples)
        dists = [self.distance_func(sample, z_gt) for sample in samples]
        for i, dist in enumerate(dists):
            if dist <= self.epsilon:
                return i + 1, dist
        
        print("Original molecule not in the epsilon neighborhood of the samples.")
        return float("inf"), float("inf")
        