import os
from pathlib import Path

# ENVIRONMENT

if os.getenv('ENV') == 'kaggle':
    DATA_DIR = Path('../input/petfinder-pawpularity-score/')
    OUT_DIR = Path('.')
else:
    PROJECT_ROOT = Path(os.getenv('PROJECT_ROOT'))
    DATA_DIR = PROJECT_ROOT / 'data'
    OUT_DIR = PROJECT_ROOT / 'model'

# DATA CONFIGS
META_COLS = eval(os.getenv('META_COLS'))
IMAGENET_MEAN = eval(os.getenv('IMAGENET_MEAN'))
IMAGENET_STD = eval(os.getenv('IMAGENET_STD'))
IMG_DIM = int(os.getenv('IMG_DIM'))
N_SPLITS = int(os.getenv('N_SPLITS'))

# DATA PATHS
TRAIN_IMG_DIR = DATA_DIR / 'train'
TEST_IMG_DIR = DATA_DIR / 'test'

TRAIN_CSV = DATA_DIR / 'train.csv'
TEST_CSV = DATA_DIR / 'train.csv'


# MODEL CONFIGS
MODEL_NAME = str(os.getenv('MODEL_NAME'))
DROPOUT = float(os.getenv('DROPOUT'))
LIN_HIDDEN = int(os.getenv('LIN_HIDDEN'))

# TRAIN CONFIGS
TRAIN_BATCH_SIZE = int(os.getenv('TRAIN_BATCH_SIZE'))
VAL_BATCH_SIZE = int(os.getenv('VAL_BATCH_SIZE'))
LR = float(os.getenv('LR'))
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS'))
NUM_VAL_PER_EPOCH = int(os.getenv('NUM_VAL_PER_EPOCH'))
REDUCE_LR_PATIENCE = int(os.getenv('REDUCE_LR_PATIENCE'))
EARLY_STOP_PATIENCE = int(os.getenv('EARLY_STOP_PATIENCE'))

# WANDB CONFIGS

WANDB_PROJECT = os.getenv('WANDB_PROJECT')
WANDB_ENTITY = os.getenv('WANDB_ENTITY')
RUN_NAME = os.getenv('RUN_NAME')
