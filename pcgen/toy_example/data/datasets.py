from pcgen.data.datasets import ConditionalDataset
from pcgen.toy_example.paths import DATA_DIR
import torch

class ToyDataset(ConditionalDataset):
    def __init__(self, path=DATA_DIR + '/data.pt'):
        data = torch.load(path)
        super(ToyDataset, self).__init__(z=data)
