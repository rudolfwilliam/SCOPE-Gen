"""These are custom functions that were not present in the DiGress codebase."""

import torch
from torch.functional import F
from rdkit import Chem
from rdkit.Chem import SanitizeFlags
from pcgen import utils


def load_model_from_ckpt_path(path, map_location="cpu"):
    import sys
    import torch
    import warnings
    from pcgen.molecules.models.molecule_generator import DiGressMoleculeGenerator
    # this is necessary because of serialization issues (super annoying, please let me know if there is a better way)
    sys.path.insert(0, find_src_path())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ckpt = torch.load(path, map_location=map_location)
        model = DiGressMoleculeGenerator(**ckpt["hyper_parameters"])
        # there are issues with the metrics of the model when moving between devices, so we remove them
        model.train_metrics = None
        model.sampling_metrics = None
        model.load_state_dict(ckpt["state_dict"])
    return model.to(map_location)

def find_src_path():
    import sys
    from pathlib import Path
    site_packages = sys.path
    env_path = Path(sys.prefix)

    # Search for the `src` directory inside the site-packages
    src_path = None
    for sp in site_packages:
        if 'site-packages' in sp and (env_path / sp).is_dir():
            src_path_candidate = env_path / sp / 'src'
            if src_path_candidate.exists() and src_path_candidate.is_dir():
                src_path = src_path_candidate
                break

    if src_path is None:
        raise FileNotFoundError("Unable to locate 'src' directory in the current conda environment.")
    
    return str(src_path)
