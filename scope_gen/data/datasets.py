from torch.utils.data import Dataset


class ConditionalDataset(Dataset):
    def __init__(self, z, cond=None):
        self.z = z
        self.cond = cond
        if cond is not None:
            assert len(z) == len(cond)
    
    def __len__(self):
        return len(self.z)
    
    def __getitem__(self, idx):
        if self.cond is not None:
            return self.z[idx], self.cond[idx]
        return self.z[idx], None
    

class CountDataset(Dataset):
    def __init__(self, y, cond=None):
        self.y = y
        self.cond = cond
        if cond is not None:
            assert len(y) == len(cond)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        if self.cond is not None:
            return self.y[idx], self.cond[idx]
        return self.y[idx], None
     