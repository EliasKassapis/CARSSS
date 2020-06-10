from pathlib import Path

from utils import constants


LIDC_PATH = Path('./data/datasets/lidc/')
CITYSCAPES_PATH = Path('./data/datasets/cityscapes/')

check_paths = [LIDC_PATH, CITYSCAPES_PATH]

# assert all([p.exists() for p in check_paths])
WRITER_DIRECTORY = "results/output"

