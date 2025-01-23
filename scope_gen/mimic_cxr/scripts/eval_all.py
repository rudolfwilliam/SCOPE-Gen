"""Script that evaluates all baselines at once."""

from scope_gen.mimic_cxr.scripts.eval_clm import eval as eval_clm
from scope_gen.mimic_cxr.scripts.eval import eval as eval_scacgen
from scope_gen.mimic_cxr.paths import CONFIG_DIR
from scope_gen.utils import load_config_from_json, load_configs_from_jsonl, set_seed
import psutil

VERBOSE = True
CLM_ONLY = False
SCACGEN_ONLY = True

def main(cfg, cfgs_scacgen, cfgs_clm, dir_):
    if VERBOSE:
        print(f"Running evaluation for {len(cfgs_scacgen)} SCAC-Gen configurations and {len(cfgs_clm)} CLM configurations.")
        print(f"Multiprocessing with {psutil.cpu_count(logical=False)} processes.")
    if not CLM_ONLY:
        for cfg_scacgen in cfgs_scacgen:
            eval_scacgen(cfg=cfg, dir_=dir_, **cfg_scacgen)
            if VERBOSE:
                print(f"Finished evaluation for SCAC-Gen configuration {cfg_scacgen}.")
    else:
        print("Skipping SCAC-Gen evaluation.")
    if not SCACGEN_ONLY:
        for cfg_clm in cfgs_clm:
            eval_clm(cfg, dir_, **cfg_clm)
            if VERBOSE:
                print(f"Finished evaluation for CLM configuration {cfg_clm}.")
    else:
        print("Skipping CLM evaluation.")


if __name__ == "__main__":
    set_seed(0)
    # load global config
    cfg = load_config_from_json(CONFIG_DIR + "/eval.json")
    # load json lines configs for both methods
    cfgs_scacgen = load_configs_from_jsonl(CONFIG_DIR + "/scac_gen.jsonl")
    cfgs_clm = load_configs_from_jsonl(CONFIG_DIR + "/clm.jsonl")
    main(cfg, cfgs_scacgen, cfgs_clm, dir_="processed")
