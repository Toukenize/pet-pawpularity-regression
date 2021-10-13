from pathlib import Path

IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD = [0.229, 0.224, 0.225]  # RGB


# Data paths
PROJECT_ROOT = Path(__file__).parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
TRAIN_IMG_DIR = DATA_DIR / 'train'
TEST_IMG_DIR = DATA_DIR / 'test'

TRAIN_CSV = DATA_DIR / 'train.csv'
TEST_CSV = DATA_DIR / 'train.csv'
