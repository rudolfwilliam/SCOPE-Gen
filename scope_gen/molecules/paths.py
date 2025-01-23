import os

DIR = os.path.relpath(os.path.dirname(__file__), ".")
DATA_DIR = os.path.join(DIR, 'data')
MODEL_DIR = os.path.join(DIR, 'models')
TRAIN_DIR = os.path.join(DIR, 'training')
OUTPUT_DIR = os.path.join(DIR, 'output')
SCRIPTS_DIR = os.path.join(DIR, 'scripts')
CONFIG_DIR = os.path.join(SCRIPTS_DIR, 'configs')
CKPT_DIR = os.path.join(MODEL_DIR, 'checkpoints')
LOG_DIR = os.path.join(TRAIN_DIR, 'logs')
TRAIN_PARAMS_DIR = os.path.join(TRAIN_DIR, 'params')
MODEL_PARAMS_DIR = os.path.join(MODEL_DIR, 'params')