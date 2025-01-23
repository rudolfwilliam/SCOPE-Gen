"""Script that evaluates all baselines at once."""

from scope_gen.cnn_dm.scripts.eval_clm import eval as eval_clm
from scope_gen.cnn_dm.scripts.eval import eval as eval_scope_gen
from scope_gen.cnn_dm.paths import CONFIG_DIR
from scope_gen.utils import load_config_from_json, load_configs_from_jsonl, set_seed
import psutil

VERBOSE = True
CLM_ONLY = False
SCOPE_GEN_ONLY = False


def main(cfg, cfgs_scope_gen, cfgs_clm, dir_):
    if VERBOSE:
        print(f"Running evaluation for {len(cfgs_scope_gen)} SCOPE-Gen configurations and {len(cfgs_clm)} CLM configurations.")
        print(f"Multiprocessing with {psutil.cpu_count(logical=False)} processes.")
    if not CLM_ONLY:
        for cfg_scope_gen in cfgs_scope_gen:
            eval_scope_gen(cfg=cfg, dir_=dir_, **cfg_scope_gen)
            if VERBOSE:
                print(f"Finished evaluation for SCOPE-Gen configuration {cfg_scope_gen}.")
    else:
        print("Skipping SCOPE-Gen evaluation.")
    if not SCOPE_GEN_ONLY:
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
    cfgs_scope_gen = load_configs_from_jsonl(CONFIG_DIR + "/scope_gen.jsonl")
    cfgs_clm = load_configs_from_jsonl(CONFIG_DIR + "/clm.jsonl")
    main(cfg, cfgs_scope_gen, cfgs_clm, dir_="processed")
