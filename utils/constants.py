from collections import OrderedDict
import cv2
import numpy as np
import torch

from models.general.data_management import DataManager

# directories
PIC_DIR = "output_pictures"
MODELS_DIR = "saved_models"
PROGRESS_DIR = "training_progress"
LOSS_DIR = "losses"
EMBED_DIR = "embedders"
GEN_DIR = "generators"
OPTIMS = "optim"
DIS_DIR = "discriminators"
PREFIX_OUTPUT = "results/output"
CODE_DIR = "codebase"
NUMPY_DIR = "numpy_arrays"
OUTPUT_DIRS = [PIC_DIR, PROGRESS_DIR, MODELS_DIR, CODE_DIR, NUMPY_DIR]

# train
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

IMSIZE = (128, 128)
# IMSIZE = (256, 256)
# IMSIZE = (256, 512)
CHANNEL_DIM = 1
LABELS_CHANNELS = 2 #2 for LIDC, #20 + 5 CITYSCAPES19, #35 CITYSCAPES35
INPUT_CHANNELS = 3
BATCH_SIZE = 2
TOTAL_G_LOSS = "TotalGeneratorLoss"
TOTAL_D_LOSS = "TotalDiscriminatorLoss"

DATA_MANAGER = DataManager(f"./{PREFIX_OUTPUT}/")

# printing
PRINTCOLOR_GREEN = '\033[92m'
PRINTCOLOR_RED = '\033[91m'
PRINTCOLOR_BOLD = '\033[1m'
PRINTCOLOR_END = '\033[0m'

# plots
MAX_ALEATORIC_GT = np.log(LABELS_CHANNELS).item()  # compute entropy of uniform categorical distribution
MAX_ALEATORIC = MAX_ALEATORIC_GT

# metrics
BEST_GED = 1
BEST_CALNET_SCORE = 10

# # dataset
INTERPOLATION = cv2.INTER_NEAREST


# LIDC
LIDC_IMSIZE = (128,128)


# CITYSCAPES35
CITYSCAPES35_IMSIZE = (1024,2048)

CITYSCAPES35_COLOUR_ENCODING = OrderedDict([
    ('unlabeled',(0, 0, 0)),
    ('ego vehicle', (0, 0, 0)),
    ('rectification border', (0, 0, 0)),
    ('out of roi', (0, 0, 0)),
    ('static', (0, 0, 0)),
    ('dynamic', (111, 74, 0)),
    ('ground', (81, 0, 81)),
    ('road', (128, 64, 128)),
    ('sidewalk', (244, 35, 232)),
    ('parking', (250, 170, 160)),
    ('rail track', (230, 150, 140)),
    ('building', (70, 70, 70)),
    ('wall', (102, 102, 156)),
    ('fence', (190, 153, 153)),
    ('guard rail', (180, 165, 180)),
    ('bridge', (150, 100, 100)),
    ('tunnel', (150, 120, 90)),
    ('pole', (153, 153, 153)),
    ('polegroup', (153, 153, 153)),
    ('traffic light', (250, 170, 30)),
    ('traffic sign', (220, 220, 0)),
    ('vegetation', (107, 142, 35)),
    ('terrain', (152, 251, 152)),
    ('sky', (70, 130, 180)),
    ('person', (220, 20, 60)),
    ('rider', (255, 0, 0)),
    ('car', (0, 0, 142)),
    ('truck', (0, 0, 70)),
    ('bus', (0, 60, 100)),
    ('caravan', (0, 0, 90)),
    ('trailer', (0, 0, 110)),
    ('train', (0, 80, 100)),
    ('motorcycle', (0, 0, 230)),
    ('bicycle', (119, 11, 32)),
    ('license plate', (0, 0, 142)),
])

CITYSCAPES35_IGNORED_CLASSES = [0,1,2,3,4,5,6,9,10,14,15,16,18,29,30,34]

CITYSCAPES35_NEW_COLOURS = [np.asarray(v) for v in CITYSCAPES35_COLOUR_ENCODING.values()]

CITYSCAPES35_N_LABELS = np.arange(len(CITYSCAPES35_NEW_COLOURS))


# CITYSCAPES19

CITYSCAPES19_IMSIZE = (1024,2048)

CITYSCAPES19_COLOUR_ENCODING = OrderedDict([
    ('road', (128, 64, 128)),
    ('sidewalk', (60, 40, 222)), # changed colour
    ('building', (70, 70, 70)),
    ('wall', (102, 102, 156)),
    ('fence', (190, 153, 153)),
    ('pole', (153, 153, 153)), #5
    ('traffic light', (250, 170, 30)),
    ('traffic sign', (220, 220, 0)),
    ('vegetation', (107, 142, 35)),
    ('terrain', (152, 251, 152)),
    ('sky', (70, 130, 180)), #10
    ('person', (220, 20, 60)),
    ('rider', (0, 0, 110)), # changed colour
    ('car', (0, 0, 142)),
    ('truck', (0, 0, 70)),
    ('bus', (0, 60, 100)), #15
    ('train', (0, 80, 100)),
    ('motorcycle', (0, 0, 230)),
    ('bicycle', (119, 11, 32)),
    ###################################### #todo added these for random class flipping
    ('sidewalk2', (219, 112, 147)),  # p=8/17 (0,80,100)
    ('person2', (0, 180, 0)),  # p=7/17
    ('car2', (255, 0, 255)),  # p=6/17
    ('vegetation2', (255, 0, 0)),  # p=5/17 (255, 0, 0)  (255,165,0) #todo change this!
    ('road2', (255, 165, 0)),  # p=4/17 (220, 20, 60)
    #######################################
    ('unlabeled',(0, 0, 0))
])

CITYSCAPES19_CLASSES = ['road','sidewalk', 'building', 'wall','fence','pole','traffic light','traffic sign','vegetation','terrain','sky','person','rider','car','truck','bus','train','motorcycle','bicycle','sidewalk2','person2','car2','vegetation2','road2','unlabeled']

CITYSCAPES19_FLIP_CLASSES = {
    "sidewalk": ("sidewalk2", 8/17),
    "person": ("person2", 7/17),
    "car": ("car2", 6/17),
    "vegetation": ("vegetation2", 5/17),
    "road": ("road2", 4/17),
}

CITYSCAPES19_DEEPFLIP = ([1, 11, 13, 8, 0], [19,20,21,22,23], [8/17, 7/17, 6/17, 5/17, 4/17]) # original classes - corresponding flip classes - probabilities
CITYSCAPES19_ROADFLIP = ([0], [23], [8/17])

CITYSCAPES19_NEW_COLOURS = [np.asarray(v) for v in CITYSCAPES19_COLOUR_ENCODING.values()]

CITYSCAPES19_N_LABELS = np.arange(len(CITYSCAPES19_NEW_COLOURS))