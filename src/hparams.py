# Supervised loss weights
import numpy as np
from tensorflow.keras.layers import LeakyReLU


def random_learning_rate(lower_bound=0.01, upper_bound=1.0) -> float:
    return np.random.uniform(lower_bound, upper_bound) * np.random.choice([1, 0.1, 0.01 ])


PROJECT_NAME = "prac-3"
GENDER = False
NORMALIZE_OUTPUT = True
EPOCHS = 10

# model 1 = baseline, 2 = baseline_attention, 3 = unet,
#       4 = residual_attention_unet, 5= inception_attention_unet,
#       6 = conv2d_attention_unet

# submodel for only model 2
# submodel 1= one_attention_output_attention, 2= one_attention_output_cnn
#          3= all_attention_output_attention, 4 = all_attention_output_cnn
MODEL_NO = 3
SUB_MODEL_NO = 2


# NOTES = ""
MODEL_NAME = ""


# **********************************

# TRAIN_CACHE_PATH = "../data/artifact/cache/train_cache"
# VAL_CACHE_PATH = "../data/artifact/cache/val_cache"


if NORMALIZE_OUTPUT:
    TARGET_VAR = "bone_age_z"
else:
    TARGET_VAR = "boneage"

# Debugging section
INIT_WB = True

# NB:
# data set variable
IMG_SIZE = 256

# ------------------------

# Network parameters


# Training parameters
START_LR = random_learning_rate()
BATCH_SIZE = 16

ALPHA = 0.1
BATCH_NORM = True
PATIENCE = 10
RECURRENT = 1
HID_ACT_FUNC = LeakyReLU(ALPHA)

# Setup dictionary with all hyperparameters
# loc_var = locals().copy()


# for key in loc_var:
#     if key[:2] != "__" and key != "torch":
#         CONFIG[key] = loc_var[key]
CONFIG = {
    "MODEL_NO": MODEL_NO,
    "SUB_MODEL_NO": SUB_MODEL_NO,
    "GENDER": GENDER,
    "TARGET_VAR": TARGET_VAR,
    "NORMALIZE_OUTPUT": NORMALIZE_OUTPUT,
    "INIT_WB": INIT_WB,
    "IMG_SIZE": IMG_SIZE,
    "START_LR": START_LR,
    "BATCH_SIZE": BATCH_SIZE,
    "EPOCHS": EPOCHS,
    "ALPHA": ALPHA,
    "BATCH_NORM": BATCH_NORM,
    "HID_ACT_FUNC": "LeakyReLU",
    "PATIENCE": PATIENCE,
}
