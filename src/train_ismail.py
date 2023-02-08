# %%
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
from models import SmallCNN, Inception
# !wandb login  # Login command for Weights and Biases library

# To disable the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ## HyperParameters

img_size = 256

# ## Load and Preprocess Input Dataset

# machine = "remote_system"
machine = "local"
# loading Data
    
### Set here the localtion of the data folder of your google drive
if machine == "remote_system":
    train_dir = "/server0/0/2022/mtirmizi/Documents/bone_data/Bone Age Training Set/"
    validation_dir = "/server0/0/2022/mtirmizi/Documents/bone_data/Bone Age Validation Set/"
    test_dir = "/server0/0/2022/mtirmizi/Documents/bone_data/Bone Age Test Set/"
else:
    train_dir = "/home/teemo/Documents/bone_data/Bone Age Training Set/"
    validation_dir = "/home/teemo/Documents/bone_data/Bone Age Validation Set/"
    test_dir = "/home/teemo/Documents/bone_data/Bone Age Test Set/"

### Train Data
train_image_dir = os.path.join( train_dir, "boneage-training-dataset")
train_df = pd.read_csv( os.path.join(train_dir,"train.csv") )

### Validation Data
validation_image_dir = os.path.join( validation_dir, "boneage-validation-dataset")
valid_df = pd.read_csv( os.path.join(validation_dir,"Validation Dataset.csv") )

### Test Data
test_image_dir = os.path.join(test_dir, "boneage-testing-dataset")
test_df = pd.read_excel(  os.path.join(test_dir, "test.xlsx"))


# Preprocess Train Dataset
train_df["male"] = train_df["male"].astype(int)

# Preprocess Validation Dataset
valid_df = valid_df.rename(columns={'Bone Age (months)': 'boneage', 'Image ID': 'id'})
valid_df["male"] = valid_df["male"].astype(int)

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

# train_df = train_df[:64]

# train_dataset = create_dataset_from_file(train_df["img_path"], train_df["male"].to_numpy().reshape(-1, 1), train_df["bone_age_z"], use_gender=False, batch_size=32)
# valid_dataset = create_dataset_from_file(valid_df["img_path"], valid_df["male"].to_numpy().reshape(-1, 1), valid_df["bone_age_z"], use_gender=False, batch_size=32)
# test_dataset = create_dataset_from_file(test_df["img_path"], test_df["male"].to_numpy().reshape(-1, 1), test_df["bone_age_z"], use_gender=False, batch_size=32)

# train_dataset_wg = create_dataset_from_file(train_df["img_path"], train_df["male"].to_numpy().reshape(-1, 1), train_df["bone_age_z"], use_gender=True, batch_size=32)
# valid_dataset_wg = create_dataset_from_file(valid_df["img_path"], valid_df["male"].to_numpy().reshape(-1, 1), valid_df["bone_age_z"], use_gender=True, batch_size=32)
# test_dataset_wg = create_dataset_from_file(test_df["img_path"], test_df["male"].to_numpy().reshape(-1, 1), test_df["bone_age_z"], use_gender=True, batch_size=32)


# %% [markdown]
# ## Model Training

# %%
def mae_in_months(x_p, y_p):
    '''function to return mae in months'''
    return mean_absolute_error((std_bone_age*x_p + mean_bone_age), (std_bone_age*y_p + mean_bone_age)) 

def random_learning_rate(lower_bound=0.01, upper_bound=1.0) -> float:
    return np.random.uniform(lower_bound, upper_bound) * np.random.choice([1, 0.1, 0.01 ])

# %% [markdown]
# ### Prepare callback functions

# %%
# lst_lrs = [0.1, 0.01, 0.001, 0.0001]
# lst_epochs = [1_0, 2_0, 3_0, 4_0]

lst_lrs = [0.01]
lst_epochs = [1_0]

# %%
# body_data = np.random.randint(10000,  size=(1280, 100))
# tags_data = np.random.randint(2,      size=(1280, 4)).astype("float32")
# np.random.random() 


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
    min_lr=1e-10,
)


# ## Base Convolution Neural Network (BaseCNN)

# for i in range(3):
#     lr = random_learning_rate()
#     batch_size = np.random.choice([8, 16, 32, 64])
#     epoch = np.random.choice([2, 3, 4 ])

#     # Set Batch Size in the datasets
#     train_dataset = create_dataset_from_file(train_df["img_path"], train_df["male"].to_numpy().reshape(-1, 1), train_df["bone_age_z"], use_gender=False, batch_size=batch_size)
#     valid_dataset = create_dataset_from_file(valid_df["img_path"], valid_df["male"].to_numpy().reshape(-1, 1), valid_df["bone_age_z"], use_gender=False, batch_size=batch_size)
#     test_dataset = create_dataset_from_file(test_df["img_path"], test_df["male"].to_numpy().reshape(-1, 1), test_df["bone_age_z"], use_gender=False, batch_size=batch_size)

#     train_dataset_wg = create_dataset_from_file(train_df["img_path"], train_df["male"].to_numpy().reshape(-1, 1), train_df["bone_age_z"], use_gender=True, batch_size=batch_size)
#     valid_dataset_wg = create_dataset_from_file(valid_df["img_path"], valid_df["male"].to_numpy().reshape(-1, 1), valid_df["bone_age_z"], use_gender=True, batch_size=batch_size)
#     test_dataset_wg = create_dataset_from_file(test_df["img_path"], test_df["male"].to_numpy().reshape(-1, 1), test_df["bone_age_z"], use_gender=True, batch_size=batch_size)

#     # Weights and Biases run initialization
#     run = wandb.init(project="jan12-run", 
#                     entity="hda-project",  # Entity is my team name on wandb website
#                     name = "BaseCNN_v3wg",
#                     config = {
#                     "architecture": "base_conv_no_gender_info",
#                     "start_lr": lr,
#                     "batch_size": batch_size
#                     })
#     # wandb.config["learning_rate"] = lr
#     # wandb.config["epochs"] = epoch
#     callbacks = [red_lr_plat, WandbCallback()]

#     input_img = tf.keras.Input(shape=(img_size, img_size, 3), name="image")
#     input_gender = tf.keras.Input(shape=(1), name="gender")

#     model = SmallCNN( input_img=input_img, input_gender=input_gender )()
#     # model = Inception((img_size, img_size, 3))()

#     optimizer = tf.keras.optimizers.Adam( lr )

#     #compile model
#     model.compile(loss = 'mse', optimizer = optimizer , metrics = [mae_in_months])

#     # Train the model
#     model.fit(train_dataset_wg,  epochs = epoch, callbacks=callbacks, validation_data=valid_dataset_wg)

#     # Tell W&B that a model run is complete
#     run.finish() 

# Save the Model
# model.save(os.path.join("..", "nn_models", "model.h5"))


# ## Inceptionv4 Neural Network (Inv4NN)

with_gender = True
for i in range(2):
    lr = random_learning_rate()
    batch_size = np.random.choice([8, 16, 32])
    epoch = np.random.choice([50, 150])

    # lr = random_learning_rate()
    # batch_size = np.random.choice([8, 16, 32, 64])
    # epoch = np.random.choice([3, 4, 5 ])

    if not with_gender:
        # Set Batch Size in the datasets
        train_dataset = create_dataset_from_file(train_df["img_path"], train_df["male"].to_numpy().reshape(-1, 1), train_df["bone_age_z"], use_gender=False, batch_size=batch_size)
        valid_dataset = create_dataset_from_file(valid_df["img_path"], valid_df["male"].to_numpy().reshape(-1, 1), valid_df["bone_age_z"], use_gender=False, batch_size=batch_size)
        test_dataset = create_dataset_from_file(test_df["img_path"], test_df["male"].to_numpy().reshape(-1, 1), test_df["bone_age_z"], use_gender=False, batch_size=batch_size)
    else:
        train_dataset_wg = create_dataset_from_file(train_df["img_path"], train_df["male"].to_numpy().reshape(-1, 1), train_df["bone_age_z"], use_gender=True, batch_size=batch_size)
        valid_dataset_wg = create_dataset_from_file(valid_df["img_path"], valid_df["male"].to_numpy().reshape(-1, 1), valid_df["bone_age_z"], use_gender=True, batch_size=batch_size)
        test_dataset_wg = create_dataset_from_file(test_df["img_path"], test_df["male"].to_numpy().reshape(-1, 1), test_df["bone_age_z"], use_gender=True, batch_size=batch_size)

    # Weights and Biases run initialization
    run = wandb.init(project="hda-final", 
                    entity="hda-project",  # Entity is my team name on wandb website
                    name = "Incenptionv4-v2-wg",
                    config = {
                    "MODEL_NAME": "base_conv_no_gender_info",
                    "START_LR": lr,
                    "BATCH_SIZE": batch_size,
                    "GENDER": with_gender
                    })
    # wandb.config["learning_rate"] = lr
    # wandb.config["epochs"] = epoch
    callbacks = [red_lr_plat, WandbCallback()]

    input_img = tf.keras.Input(shape=(img_size, img_size, 3), name="image")
    input_gender = tf.keras.Input(shape=(1), name="gender")

    optimizer = tf.keras.optimizers.Adam( lr )

    if not with_gender:
        model = Inception((img_size, img_size, 3))()

        #compile model
        model.compile(loss = 'mse', optimizer = optimizer, metrics = [mae_in_months])

        # Train the model
        model.fit(train_dataset,  epochs = epoch, callbacks=callbacks, validation_data=valid_dataset)

    else:
        model = Inception((img_size, img_size, 3), input_gender=input_gender)()

        #compile model
        model.compile(loss = 'mse', optimizer = optimizer, metrics = [mae_in_months])

        # Train the model
        model.fit(train_dataset_wg,  epochs = epoch, callbacks=callbacks, validation_data=valid_dataset_wg)


    # Tell W&B that a model run is complete
    run.finish() 



