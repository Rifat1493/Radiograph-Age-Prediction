# Supervised loss weights
from tensorflow.keras.layers import LeakyReLU

PROJECT_NAME = "prac-2"
# model 1 = baseline, 2 = baseline_attention, 3 = unet,
#       4 = residual_attention_unet, 5= inception_attention_unet,
#       6 = conv2d_attention_unet

# submodel for only model 2
# submodel 1= one_attention_output_attention, 2= one_attention_output_cnn
#          3= all_attention_output_attention, 4 = all_attention_output_cnn
MODEL_NO = 1
SUB_MODEL_NO = 2

GENDER = False
# NOTES = ""
MODEL_NAME = ""


# **********************************

# TRAIN_CACHE_PATH = "../data/artifact/cache/train_cache"
# VAL_CACHE_PATH = "../data/artifact/cache/val_cache"


TARGET_VAR = "boneage"

if TARGET_VAR == "boneage":
    NORMALIZE_OUTPUT = False
else:
    NORMALIZE_OUTPUT = True

# Debugging section
INIT_WB = True

# NB:
# data set variable
IMG_SIZE = 256

# ------------------------

# Network parameters


# Training parameters
LR = 1e-4
BATCH_SIZE = 8
EPOCHS = 10
ALPHA = 0.1
BATCH_NORM = False
PATIENCE = 7
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
    "LR": LR,
    "BATCH_SIZE": BATCH_SIZE,
    "EPOCHS": EPOCHS,
    "ALPHA": ALPHA,
    "BATCH_NORM": BATCH_NORM,
    "HID_ACT_FUNC": "LeakyReLU",
    "PATIENCE": PATIENCE
}
