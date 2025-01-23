import os

DIR = os.path.relpath(os.path.dirname(__file__), ".")
DATA_DIR = os.path.join(DIR, 'data')
SCRIPTS_DIR = os.path.join(DIR, 'scripts')
CONFIG_DIR = os.path.join(SCRIPTS_DIR, 'configs')