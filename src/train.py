import math
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.metrics import mean_absolute_error

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import wandb
from hparams import *
from models import *
from utils import create_dataset_from_file, set_seeds, train_model

set_seeds(42)
warnings.filterwarnings("ignore")
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 1 for run in gpu -1 for run in cpu
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TF_GPU_ALLOCATOR'] ='cuda_malloc_async'


# loading dataframes
train_df = pd.read_csv("data/Bone Age Training Set/train.csv")
df_test = pd.read_excel("data/Bone Age Test Set/test.xlsx")

# appending file extension to id column for both training and testing dataframes
train_df["id"] = train_df["id"].apply(lambda x: str(x) + ".png")
df_test["Case ID"] = df_test["Case ID"].apply(lambda x: str(x) + ".png")
df_test.rename(
    columns={"Ground truth bone age (months)": "boneage", "Sex": "gender"}, inplace=True
)

train_df["img_path"] = train_df["id"].apply(
    lambda x: "data/Bone Age Training Set/boneage-training-dataset/" + str(x)
)
df_test["img_path"] = df_test["Case ID"].apply(
    lambda x: "data/Bone Age Test Set/boneage-testing-dataset/" + str(x)
)

train_df = train_df.head(200)

train_df["gender"] = train_df["male"].apply(lambda x: "male" if x else "female")

train_df["gender"].replace(["male", "female"], [1, 0], inplace=True)
df_test["gender"].replace(["M", "F"], [0, 1], inplace=True)


# mean age is
mean_bone_age = train_df["boneage"].mean()
# standard deviation of boneage
std_bone_age = train_df["boneage"].std()

# using z score for the training
train_df["bone_age_z"] = (train_df["boneage"] - mean_bone_age) / (std_bone_age)

# splitting train dataframe into traininng and validation dataframes
df_train, df_valid = train_test_split(train_df, test_size=0.2, random_state=0)


train_steps = int(np.ceil(len(df_train) / hparams.BATCH_SIZE))
val_steps = int(np.ceil(len(df_valid) / hparams.BATCH_SIZE))
train_dataset = create_dataset_from_file(
    df_train["img_path"],
    df_train["gender"].to_numpy().reshape(-1, 1),
    df_train[hparams.TARGET_VAR].to_numpy().reshape(-1, 1),
)
val_dataset = create_dataset_from_file(
    df_valid["img_path"],
    df_valid["gender"].to_numpy().reshape(-1, 1),
    df_valid[hparams.TARGET_VAR].to_numpy().reshape(-1, 1),
)
test_dataset = create_dataset_from_file(
    df_test["img_path"],
    df_test["gender"].to_numpy().reshape(-1, 1),
    df_test[hparams.TARGET_VAR].to_numpy().reshape(-1, 1),
)


def mae_in_months(x_p, y_p):
    """function to return mae in months"""
    return mean_absolute_error(
        (std_bone_age * x_p + mean_bone_age), (std_bone_age * y_p + mean_bone_age)
    )


for i in [1, [2, 1], [2, 2], [2, 3], [2, 4], 3, 4, 5, 6]:
    # MODEL_NO = i[0]
    if type(i) == list:
        hparams.MODEL_NO = i[0]
        hparams.SUB_MODEL_NO = i[1]
    else:
        hparams.MODEL_NO = i


    if hparams.NORMALIZE_OUTPUT == True:
        metric = ["mae_in_months"]
    else:
        metric = ["mse"]

    if hparams.MODEL_NO == 1:
        hparams.MODEL_NAME = "baseline"
        model = BaselineCnn.baseline_cnn()

    elif hparams.MODEL_NO == 2:
        hparams.MODEL_NAME = "baseline_attention_" + str(hparams.SUB_MODEL_NO)
        model = BaselineCnnAttention.baseline_cnn_attention(hparams.SUB_MODEL_NO)

    elif hparams.MODEL_NO == 3:
        model = Unet.unet()
        hparams.MODEL_NAME = "unet"

    elif hparams.MODEL_NO == 4:
        model = ResidualAttentionUnet.residual_attention_unet()
        hparams.MODEL_NAME = "residual_attention_unet"
    elif hparams.MODEL_NO == 5:
        model = InceptionAttentionUnet.inception_attention_unet()
        hparams.MODEL_NAME = "inception_attention_unet"
    elif hparams.MODEL_NO == 6:
        model = CnnAttentionUnet.cnn_attention_unet()
        hparams.MODEL_NAME = "cnn_attention_unet"

    model.compile(loss="mse", optimizer="adam", metrics=metric)
    wandb.init(
        project=hparams.PROJECT_NAME,
        entity="hda-project",
        name=hparams.MODEL_NAME
        # notes=hparams.NOTES
    )
    wandb.config.update(hparams.CONFIG)

    if hparams.GENDER:
        hparams.MODEL_NAME = hparams.MODEL_NAME + "_gender"

    history = train_model(model, train_dataset, val_dataset, train_steps, val_steps)

    wandb.config.update({"MODEL_NAME": hparams.MODEL_NAME})

    art = wandb.Artifact(hparams.MODEL_NAME + "_best_model", type="model")
    art.add_file("data/artifact/" + hparams.MODEL_NAME + ".h5")
    wandb.log_artifact(art)

    test_y = df_test[hparams.TARGET_VAR].to_numpy()
    pred_y = model.predict(test_dataset)
    mse_value = mean_squared_error(test_y, pred_y)
    wandb.log({"test_mse": mse_value})
    wandb.finish()
