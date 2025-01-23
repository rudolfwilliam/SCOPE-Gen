

def split_data_idxs(n, split_ratios):
    split_idxs = []
    start_idx = 0
    for ratio in split_ratios:
        end_idx = start_idx + int(ratio * n)
        split_idxs.append(list(range(start_idx, end_idx)))
        start_idx = end_idx
    return split_idxs
