import json
import argparse
import os

from scope_gen.scripts.eval import eval
from scope_gen.utils import load_config_from_json, set_seed
from scope_gen.cnn_dm.paths import CONFIG_DIR, DATA_DIR


DEBUG = False
VERBOSE = True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run the evaluation module with given parameters")
    parser.add_argument('--config', type=str, default=os.path.join(CONFIG_DIR, "eval.json"), help='Path to configuration JSON file')
    parser.add_argument('--name', type=str, default="ourmethod{}", help='Name of the method')
    parser.add_argument('--dir', type=str, default="processed", help='Directory for processing')
    parser.add_argument('--score', type=str, default="count", help='Score type')
    parser.add_argument('--custom_path', type=str, default=None, help='Custom path to storing the result')
    parser.add_argument('--return_std_coverages', type=bool, default=False, help='Return standard deviations of coverages')
    parser.add_argument('--stages', nargs='+', default=["generation", "diversity", "quality"], help='List of stages to process')
    def parse_alpha_params(s):
        return json.loads(s)
    parser.add_argument('--alpha_params', type=parse_alpha_params, default=None, help='Dictionary of alpha parameters M and parts. Parts is a list of \
                        integers of length num_steps that must sum up to M.')
    args = parser.parse_args()

    set_seed(0)
    cfg = load_config_from_json(args.config)  # Use args.config instead of hardcoded path

    eval(
        cfg,
        dir_=args.dir,
        name=args.name,
        score=args.score,
        stages=args.stages,
        data_dir=DATA_DIR,
        return_std_coverages=args.return_std_coverages,
        custom_path=args.custom_path,
        alpha_params=args.alpha_params,
        debug=DEBUG,
        verbose=VERBOSE
    )
    