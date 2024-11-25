
def set_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def override(func):
    # does nothing, just for readability
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper

def load_config_from_json(json_file):
    import json
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config

def load_configs_from_jsonl(file_path):
    import json
    configurations = []
    with open(file_path, 'r') as file:
        for line in file:
            # Parse the JSON line into a dictionary
            configuration = json.loads(line.strip())
            configurations.append(configuration)
    return configurations

def flatten_list(list):
     return sum(list, [])

def slice_dict(d, idx):
    return {key: value[:(idx + 1)] if len(value.shape) == 1 else value[:(idx + 1), :(idx + 1)]
            for key, value in d.items()}

def order_dict(d, idxs):
    import numpy as np
    return {key: value[idxs] if len(value.shape) == 1 else value[np.ix_(idxs, idxs)]
            for key, value in d.items()}

def store_results(data_dir, name, alpha, score, coverages, sizes, first_adms, times, data_set_size, type="eval_results", std_coverages=None):
    import os
    import pickle
    alpha_str = str(alpha).replace(".", "")
    if type is None:
        type = ""
    path = os.path.join(data_dir, type, name)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, f"{score}_{data_set_size}_{alpha_str}.pkl"), 'wb') as file:
        pickle.dump({"coverages" : coverages, "sizes" : sizes, "adm_counts" : first_adms, "times" : times, "std_coverages" : std_coverages}, file)

def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)
        