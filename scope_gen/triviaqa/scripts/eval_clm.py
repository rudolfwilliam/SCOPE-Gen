import numpy as np
import argparse
import os

from scope_gen.utils import load_config_from_json, set_seed
from scope_gen.triviaqa.paths import CONFIG_DIR, DATA_DIR
from scope_gen.baselines.clm.eval import eval


USE_LAMBDA_1 = False # similarity
USE_LAMBDA_2 = True # quality
ALT_LAMBDA_1 = 0.5
ALT_LAMBDA_2 = -np.inf

DEBUG = False 
VERBOSE = True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run the evaluation module with given parameters")
    parser.add_argument('--config', type=str, default=os.path.join("CONFIG_DIR", "eval.json"), help='Path to configuration JSON file')
    parser.add_argument('--name', type=str, default="CLM", help='name of the method')
    parser.add_argument('--dir', type=str, default="processed", help='Directory for processing')
    parser.add_argument('--score', type=str, default="count", help='Score type')
    parser.add_argument('--reduced_max', type=int, default=20)
    args = parser.parse_args()

    set_seed(0)
    cfg = load_config_from_json(os.path.join(CONFIG_DIR, "eval.json"))

    eval(
         cfg, 
         dir_=args.dir, 
         name=args.name, 
         score=args.score,
         data_dir=DATA_DIR,
         use_lambda_1=USE_LAMBDA_1,
         use_lambda_2=USE_LAMBDA_2,
         reduced_max=args.reduced_max,
         verbose=VERBOSE,
         debug=DEBUG
         )
