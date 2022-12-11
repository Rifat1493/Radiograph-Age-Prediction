# Supervised loss weights

MODEL_NAME = "baseline_encoder_model"
PROJECT_NAME = "prac-2"

# **********************************

TRAIN_CACHE_PATH = "data/artifact/cache/train_cache"
VAL_CACHE_PATH = "data/artifact/cache/val_cache"

# DEVICE = "cpu"

# NB:
# data set variable
IMG_SIZE = 100

# ------------------------

# Network parameters


# Training parameters
LR = 1e-4
BATCH_SIZE = 32
EPOCHS = 2

# Setup dictionary with all hyperparameters
loc_var = locals().copy()
CONFIG = {}

for key in loc_var:
    if key[:2] != "__" and key != "torch":
        CONFIG[key] = loc_var[key]
