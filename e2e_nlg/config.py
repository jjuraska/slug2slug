import os


# Directory paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EVAL_DIR = os.path.join(ROOT_DIR, 'eval')
METRICS_DIR = os.path.join(ROOT_DIR, 'metrics')
MODEL_DIR = os.path.join(ROOT_DIR, 'model')
PREDICTIONS_DIR = os.path.join(ROOT_DIR, 'predictions')
SLOT_ALIGNER_DIR = os.path.join(ROOT_DIR, 'slot_aligner')
T2T_DIR = os.path.join(ROOT_DIR, 't2t')
TOOLS_DIR = os.path.join(ROOT_DIR, 'tools')
TTEST_DIR = os.path.join(ROOT_DIR, 'ttest')
TTEST_DATA_DIR = os.path.join(ROOT_DIR, 'ttest', 'data')
TTEST_SCORES_DIR = os.path.join(ROOT_DIR, 'ttest', 'scores')

# Dataset paths
E2E_DATA_DIR = os.path.join(DATA_DIR, 'rest_e2e')
TV_DATA_DIR = os.path.join(DATA_DIR, 'tv')
LAPTOP_DATA_DIR = os.path.join(DATA_DIR, 'laptop')
HOTEL_DATA_DIR = os.path.join(DATA_DIR, 'hotel')

# Script paths
METRICS_SCRIPT_PATH = os.path.join(METRICS_DIR, 'measure_scores.py')

# Constants
EMPH_TOKEN = '<!emph>'
CONTRAST_TOKEN = '<!contrast>'
CONCESSION_TOKEN = '<!concession>'
