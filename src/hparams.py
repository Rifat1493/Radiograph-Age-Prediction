# Supervised loss weights

MODEL_NAME = "baseline_encoder_model"
PROJECT_NAME = "prac-2"

# **********************************

# TRAIN_CACHE_PATH = "../data/artifact/cache/train_cache"
# VAL_CACHE_PATH = "../data/artifact/cache/val_cache"


TARGET_VAR = "boneage"

if TARGET_VAR == "boneage":
    NORMALIZE_OUTPUT = False
else:
    NORMALIZE_OUTPUT = True

# Debugging section
INIT_WB = False

# NB:
# data set variable
IMG_SIZE = 256

# ------------------------

# Network parameters


# Training parameters
LR = 1e-4
BATCH_SIZE = 32
EPOCHS = 15
ALPHA = 0.1
BATCH_NORM = False

# Setup dictionary with all hyperparameters
loc_var = locals().copy()
CONFIG = {}

for key in loc_var:
    if key[:2] != "__" and key != "torch":
        CONFIG[key] = loc_var[key]
