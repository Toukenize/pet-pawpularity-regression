from pathlib import Path

# DATA CONFIGS
META_COLS = [
    'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
    'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'
]
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD = [0.229, 0.224, 0.225]  # RGB
IMG_DIM = 256
TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = TRAIN_BATCH_SIZE * 4
N_SPLITS = 5

# DATA PATHS
PROJECT_ROOT = Path(__file__).parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
TRAIN_IMG_DIR = DATA_DIR / 'train'
TEST_IMG_DIR = DATA_DIR / 'test'

TRAIN_CSV = DATA_DIR / 'train.csv'
TEST_CSV = DATA_DIR / 'train.csv'

OUT_DIR = PROJECT_ROOT / 'model'

# MODEL_CONFIGS
MODEL_NAME = 'resnet18'
DROPOUT = 0.15
LIN_HIDDEN = 128
LR = 1e-4
NUM_EPOCHS = 5
NUM_WARMUP_EPOCHS = 2
NUM_COS_CYCLE = 0.4
