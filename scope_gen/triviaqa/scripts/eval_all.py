"""Script that evaluates all baselines at once."""

from scope_gen.scripts.eval import eval_all
from scope_gen.triviaqa.paths import CONFIG_DIR, DATA_DIR
from scope_gen.utils import load_config_from_json, load_configs_from_jsonl, set_seed


VERBOSE = True
CLM_ONLY = False
SCOPE_GEN_ONLY = False


if __name__ == "__main__":
    set_seed(0)
    # load global config
    cfg = load_config_from_json(CONFIG_DIR + "/eval.json")
    # load json lines configs for both methods
    cfgs_scope_gen = load_configs_from_jsonl(CONFIG_DIR + "/scope_gen.jsonl")
    cfgs_clm = load_configs_from_jsonl(CONFIG_DIR + "/clm.jsonl")
    
    eval_all(cfg, 
             cfgs_scope_gen,
             cfgs_clm, 
             dir_="processed",
             verbose=VERBOSE,
             clm_only=CLM_ONLY,
             data_dir=DATA_DIR,
             scope_gen_only=SCOPE_GEN_ONLY
            )
