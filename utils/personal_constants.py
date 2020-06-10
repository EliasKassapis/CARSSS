from pathlib import Path

from utils import constants


# LIDC_PATH = Path('/home/kassapis/Desktop/CAR/data/datasets/lidc/')
LIDC_PATH = Path('./data/datasets/lidc/')


# LIDC_RAW_PATH = Path('/home/kassapis/projects/MSc_Thesis/data/datasets/LIDC/')
# LIDC_RAW_PATH = Path('/home/kassapis/Desktop/MSc_Thesis/data/datasets/LIDC/')
# LIDC_RAW_PATH = Path('./data/datasets/LIDC/')


# CITYSCAPES_PATH = Path('/home/kassapis/Desktop/CAR/data/datasets/cityscapes/')
CITYSCAPES_PATH = Path('./data/datasets/cityscapes/')


# CITYSCAPES_PATH = Path('/home/kassapis/projects/MSc_Thesis/data/datasets/Cityscapes')
# CITYSCAPES_PATH = Path('/home/kassapis/Desktop/MSc_Thesis/data/datasets/Cityscapes')
# CITYSCAPES_PATH = Path("./data/datasets/Cityscapes")

check_paths = [LIDC_PATH, CITYSCAPES_PATH]

# assert all([p.exists() for p in check_paths])
WRITER_DIRECTORY = "results/output"

