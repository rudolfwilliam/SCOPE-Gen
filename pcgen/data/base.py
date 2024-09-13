"""from torch.utils.data import Subset

def split_data_set(total_data, split_ratio=[0.8, 0.1, 0.1]):
    split_idx_1 = int(split_ratio[0] * len(total_data))
    split_idx_2 = int((split_ratio[0] + split_ratio[1]) * len(total_data))

    # Create subsets
    idxs_1 = list(range(split_idx_1))
    idxs_2 = list(range(split_idx_1, split_idx_2))
    idxs_3 = list(range(split_idx_2, len(total_data)))

    set_1 = Subset(total_data, idxs_1)
    set_2 = Subset(total_data, idxs_2)
    set_3 = Subset(total_data, idxs_3)

    return set_1, set_2, set_3"""

def split_data_idxs(n, split_ratios):
    split_idxs = []
    start_idx = 0
    for ratio in split_ratios:
        end_idx = start_idx + int(ratio * n)
        split_idxs.append(list(range(start_idx, end_idx)))
        start_idx = end_idx
    return split_idxs
