import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras.metrics import mean_absolute_error
import datetime, os
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg

import wandb
from wandb.keras import WandbCallback

#library required for image preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from utils import create_dataset_from_file
from models import SmallCNN
# !wandb login  # Login command for Weights and Biases library

# %% [markdown]
# ## HyperParameters

# %%
img_size = 256

# %% [markdown]
# ## Load and Preprocess Input Dataset

# %%
machine = "local"
# loading Data
if machine == "Google-Drive":
    from google.colab import drive
    drive.mount('/content/drive/')
    
### Set here the localtion of the data folder of your google drive
if machine == "Google-Drive":
    train_dir = "/content/drive/MyDrive/BDMA/3-semester/human_data_analysis/Bone Age 1k Set/"
    train_df = pd.read_csv( os.path.join(train_dir,"train.csv") )
    
else:
    train_dir = "/home/teemo/Documents/bone_data/Bone Age Training Set/"
    train_image_dir = os.path.join( train_dir, "boneage-training-dataset")
    train_df = pd.read_csv( os.path.join(train_dir,"train.csv") )

# Preprocess Train Dataset
train_df["male"] = train_df["male"].astype(int)


### Validation Data
validation_dir = "/home/teemo/Documents/bone_data/Bone Age Validation Set/"
validation_image_dir = os.path.join( validation_dir, "boneage-validation-dataset")
valid_df = pd.read_csv( os.path.join(validation_dir,"Validation Dataset.csv") )

# Preprocess Validation Dataset
valid_df = valid_df.rename(columns={'Bone Age (months)': 'boneage', 'Image ID': 'id'})
valid_df["male"] = valid_df["male"].astype(int)


### Test Data
test_dir = "/home/teemo/Documents/bone_data/Bone Age Test Set/"
test_image_dir = os.path.join(test_dir, "boneage-testing-dataset")
test_df = pd.read_csv(  os.path.join(test_dir, "test.csv"))

# Preprocess Test Dataset
test_df = test_df.rename(columns={'Ground truth bone age (months)': 'boneage', 'Case ID': 'id'})
test_df["male"] = test_df['Sex'].replace(['M', 'F'], [1, 0])
test_df = test_df.drop(columns=["Sex"])


# Appending file extension to id column for both training and testing dataframes
train_df['id'] = train_df['id'].apply(lambda x: str(x) + '.png')
valid_df['id'] = valid_df['id'].apply(lambda x: str(x) + '.png')
test_df['id'] = test_df['id'].apply(lambda x: str(x) + '.png') 

# Create Image paths. Will be needed in tensorflow Dataset API
train_df['img_path'] = train_df['id'].apply(lambda x: os.path.join(train_image_dir, str(x)) )
valid_df['img_path'] = valid_df['id'].apply(lambda x: os.path.join(validation_image_dir, str(x)) )
test_df['img_path'] = test_df['id'].apply(lambda x: os.path.join(test_image_dir, str(x)) )

#mean age is
mean_bone_age = train_df['boneage'].mean()

#standard deviation of boneage
std_bone_age = train_df['boneage'].std()

#models perform better when features are normalised to have zero mean and unity standard deviation
#using z score for the training
train_df.loc[:, 'bone_age_z'] = (train_df['boneage'] - mean_bone_age) / std_bone_age

# Similarly z score for Validation & testing data
valid_df.loc[:, "bone_age_z"] = (valid_df["boneage"] - mean_bone_age) / std_bone_age
test_df.loc[:, 'bone_age_z'] = (test_df['boneage'] - mean_bone_age) / std_bone_age

train_df = train_df[:64]

train_dataset = create_dataset_from_file(train_df["img_path"], train_df["male"].to_numpy().reshape(-1, 1), train_df["bone_age_z"], use_gender=False, batch_size=32)
valid_dataset = create_dataset_from_file(valid_df["img_path"], valid_df["male"].to_numpy().reshape(-1, 1), valid_df["bone_age_z"], use_gender=False, batch_size=32)
test_dataset = create_dataset_from_file(test_df["img_path"], test_df["male"].to_numpy().reshape(-1, 1), test_df["bone_age_z"], use_gender=False, batch_size=32)

train_dataset_wg = create_dataset_from_file(train_df["img_path"], train_df["male"].to_numpy().reshape(-1, 1), train_df["bone_age_z"], use_gender=True, batch_size=32)
valid_dataset_wg = create_dataset_from_file(valid_df["img_path"], valid_df["male"].to_numpy().reshape(-1, 1), valid_df["bone_age_z"], use_gender=True, batch_size=32)
test_dataset_wg = create_dataset_from_file(test_df["img_path"], test_df["male"].to_numpy().reshape(-1, 1), test_df["bone_age_z"], use_gender=True, batch_size=32)


# %% [markdown]
# ## Model Training

# %%
def mae_in_months(x_p, y_p):
    '''function to return mae in months'''
    return mean_absolute_error((std_bone_age*x_p + mean_bone_age), (std_bone_age*y_p + mean_bone_age)) 

# %% [markdown]
# ### Prepare callback functions

# %%
# lst_lrs = [0.1, 0.01, 0.001, 0.0001]
# lst_epochs = [1_0, 2_0, 3_0, 4_0]

lst_lrs = [0.1, 0.01]
lst_epochs = [1_0, 2_0]

# %%
# reduce lr on plateau
red_lr_plat = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=10,
    verbose=0,
    mode="min",
    min_delta=0.0001,
    cooldown=0,
    min_lr=0,
)

# %%
for lr in lst_lrs:
    for epoch in lst_epochs:
        
        # Weights and Biases run initialization
        run = wandb.init(project="jan12-run", 
                entity="hda-project",  # Entity is my team name on wandb website
                name = "BaseCNN",
                config = {
                  "architecture": "base_conv_no_gender_info",
                  "start_lr": lr,
                  "batch_size": 32
                })
        # wandb.config["learning_rate"] = lr
        # wandb.config["epochs"] = epoch
        callbacks = [red_lr_plat, WandbCallback()]

        # set WandB config

        input_img = tf.keras.Input(shape=(img_size, img_size, 3), name="image")

        input_gender = tf.keras.Input(shape=(1), name="gender")

        model = SmallCNN( input_img=input_img, input_gender=input_gender )()

        optimizer = tf.keras.optimizers.Adam( lr )

        #compile model
        model.compile(loss = 'mse', optimizer = optimizer , metrics = [mae_in_months])

        # Train the model
        model.fit(train_dataset_wg,  epochs = epoch, callbacks=callbacks, validation_data=valid_dataset_wg)

        # Tell W&B that a model run is complete
        run.finish() 

# Save the Model
# model.save(os.path.join("..", "nn_models", "model.h5"))



