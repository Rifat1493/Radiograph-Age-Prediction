import torch

DATA_PATH = "data/single_target"
GEN_DATA_PATH = "data/generated_dataset"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"

# NB:
# data set variable
IMG_SIZE = 100

# ------------------------

RANDOM_JITTER = True
RANDOM_SUBSAMPLE = False

# Network parameters


# Training parameters
LR = 1e-4
BATCH_SIZE = 32

# b1 and b2 are adam specific parameters
B1 = 0.9
B2 = 0.99

# regularization for encoder
L2_REG = 1e-4

TRAIN_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
EPOCHS = 80
BATCH_SIZE = 32
SHUFFLE = True

# Prior style distribution
PRIOR_MU = 0.0
PRIOR_STD = 1.0
GP_WEIGHT = 15

# Supervised loss weights
CE_W = 1
TRIPLET_W = 1
TRIPLET_MARGIN = 0.1
ADV_WEIGHT = 1

MODEL_NAME = "semisup_aae32"
NOTES = """Divide by std"""
CHECKPOINT_FREQUENCY = 20


MY_TEST = "yes"

# every x epochs there is a supervised/unsupervised step
SUPERVISION_FREQUENCY = 2

# Setup dictionary with all hyperparameters
loc_var = locals().copy()
CONFIG = {}

for key in loc_var:
    if key[:2] != "__" and key != "torch":
        CONFIG[key] = loc_var[key]
