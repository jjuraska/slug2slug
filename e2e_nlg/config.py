import os


# Directory paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EVAL_DIR = os.path.join(ROOT_DIR, 'eval')
METRICS_DIR = os.path.join(ROOT_DIR, 'metrics')
MODEL_DIR = os.path.join(ROOT_DIR, 'model')
PREDICTIONS_DIR = os.path.join(ROOT_DIR, 'predictions')
TOOLS_DIR = os.path.join(ROOT_DIR, 'tools')
TTEST_DIR = os.path.join(ROOT_DIR, 'ttest')
TTEST_DATA_DIR = os.path.join(ROOT_DIR, 'ttest', 'data')
TTEST_SCORES_DIR = os.path.join(ROOT_DIR, 'ttest', 'scores')

# Script paths
METRICS_SCRIPT_PATH = os.path.join(METRICS_DIR, 'measure_scores.py')
