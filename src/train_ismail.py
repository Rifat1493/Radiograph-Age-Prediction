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
IMG_SIZE = 256

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

train_df = train_df[:64]
valid_df = valid_df[:64]

# train_dataset = create_dataset_from_file(train_df["img_path"], train_df["male"].to_numpy().reshape(-1, 1), train_df["bone_age_z"], use_gender=False, batch_size=32)
# valid_dataset = create_dataset_from_file(valid_df["img_path"], valid_df["male"].to_numpy().reshape(-1, 1), valid_df["bone_age_z"], use_gender=False, batch_size=32)
# test_dataset = create_dataset_from_file(test_df["img_path"], test_df["male"].to_numpy().reshape(-1, 1), test_df["bone_age_z"], use_gender=False, batch_size=32)

# train_dataset_wg = create_dataset_from_file(train_df["img_path"], train_df["male"].to_numpy().reshape(-1, 1), train_df["bone_age_z"], use_gender=True, batch_size=32)
# valid_dataset_wg = create_dataset_from_file(valid_df["img_path"], valid_df["male"].to_numpy().reshape(-1, 1), valid_df["bone_age_z"], use_gender=True, batch_size=32)
# test_dataset_wg = create_dataset_from_file(test_df["img_path"], test_df["male"].to_numpy().reshape(-1, 1), test_df["bone_age_z"], use_gender=True, batch_size=32)

# Helper Functions
def mae_in_months(x_p, y_p):
    '''function to return mae in months'''
    return mean_absolute_error((std_bone_age*x_p + mean_bone_age), (std_bone_age*y_p + mean_bone_age)) 

def random_learning_rate(lower_bound=0.01, upper_bound=1.0) -> float:
    return np.random.uniform(lower_bound, upper_bound) * np.random.choice([1, 0.1, 0.01 ])

def tf_dataset_calculate_mae_in_months(tf_dataset=None):
    pred_y = np.array([])
    test_y = np.array([])
    for xray_batch in tf_dataset:
        # In xray_batch is a tuple, 1st element is features, 2nd element is the label or target
        yhat = model.predict(xray_batch[0]).flatten()
        pred_y = np.append(pred_y, yhat)
        y = xray_batch[1].numpy()
        test_y = np.append(test_y, y)
    mae = mae_in_months(pred_y, test_y)
    #     break
    # print(f"pred_y: {len(pred_y)}")
    # print(f"test_y: {len(test_y)}")
    return mae.numpy()


### Prepare callback functions

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

# model checkpoint
mc = ModelCheckpoint(
    "data/artifact/" + "model" + ".h5",
    monitor="val_loss",
    mode="min",
    save_best_only=True,
)

### Model Training

## Base Convolution Neural Network (BaseCNN)

print("Running Baseline-CNN")
with_gender = True
lr = random_learning_rate()
batch_size = np.random.choice([32])
epoch = np.random.choice([2])
for i in range(2):
    if i == 1:
        with_gender = False

    # Set Batch Size in the datasets
    if not with_gender:
        train_dataset = create_dataset_from_file(train_df["img_path"], train_df["male"].to_numpy().reshape(-1, 1), train_df["bone_age_z"], use_gender=False, batch_size=batch_size, img_size=img_size)
        valid_dataset = create_dataset_from_file(valid_df["img_path"], valid_df["male"].to_numpy().reshape(-1, 1), valid_df["bone_age_z"], use_gender=False, batch_size=batch_size, img_size=img_size)
        test_dataset = create_dataset_from_file(test_df["img_path"], test_df["male"].to_numpy().reshape(-1, 1), test_df["bone_age_z"], use_gender=False, batch_size=batch_size, img_size=img_size)
    else:
        train_dataset_wg = create_dataset_from_file(train_df["img_path"], train_df["male"].to_numpy().reshape(-1, 1), train_df["bone_age_z"], use_gender=True, batch_size=batch_size, img_size=img_size)
        valid_dataset_wg = create_dataset_from_file(valid_df["img_path"], valid_df["male"].to_numpy().reshape(-1, 1), valid_df["bone_age_z"], use_gender=True, batch_size=batch_size, img_size=img_size)
        test_dataset_wg = create_dataset_from_file(test_df["img_path"], test_df["male"].to_numpy().reshape(-1, 1), test_df["bone_age_z"], use_gender=True, batch_size=batch_size, img_size=img_size)

    # Weights and Biases run initialization
    run = wandb.init(project="hda-final", 
                    entity="hda-project",  # Entity is my team name on wandb website
                    name = f"CNN-DA-v2-g-{with_gender}",
                    config = {
                    "MODEL_NAME": "CNN-v2",
                    "START_LR": lr,
                    "BATCH_SIZE": batch_size,
                    "IMG_SIZE": img_size,
                    "GENDER": with_gender
                    })

    callbacks = [red_lr_plat, WandbCallback()]

    optimizer = tf.keras.optimizers.Adam( lr )

    input_img = tf.keras.Input(shape=(img_size, img_size, 3), name="image")
    if not with_gender:
        model = SmallCNN(input_img=input_img)()

        #compile model
        model.compile(loss = 'mse', optimizer = optimizer, metrics = [mae_in_months])

        # Train the model
        model.fit(train_dataset,  epochs = epoch, callbacks=callbacks, validation_data=valid_dataset)
        
        # predictions on test dataset
        test_mae = tf_dataset_calculate_mae_in_months(test_dataset)

    else:
        input_gender = tf.keras.Input(shape=(1), name="gender")
        model = SmallCNN( input_img=input_img, input_gender=input_gender )()

        #compile model
        model.compile(loss = 'mse', optimizer = optimizer, metrics = [mae_in_months])

        # Train the model
        model.fit(train_dataset_wg,  epochs = epoch, callbacks=callbacks, validation_data=valid_dataset_wg)

        # predictions on test dataset
        test_mae = tf_dataset_calculate_mae_in_months(test_dataset_wg)

    # # wandb automatically saves the model
    # art = wandb.Artifact(f"model-{run.name}-h5", type="model")
    # art.add_file(f"{run.dir}/model-best.h5")
    # wandb.log_artifact(art)

    # Log Performance of the test dataset
    wandb.log({"test_mae_in_months": test_mae})

    # Tell W&B that a model run is complete
    run.finish() 


# Save the Model
# model.save(os.path.join("..", "nn_models", "model.h5"))


# ## Inceptionv4 Neural Network (Inv4NN)

with_gender = True
for i in range(0):
    if i == 1:
        with_gender = False
    lr = random_learning_rate()
    batch_size = np.random.choice([4])
    epoch = np.random.choice([5])

    if not with_gender:
        # Set Batch Size in the datasets
        train_dataset = create_dataset_from_file(train_df["img_path"], train_df["male"].to_numpy().reshape(-1, 1), train_df["bone_age_z"], use_gender=False, batch_size=batch_size, img_size=img_size)
        valid_dataset = create_dataset_from_file(valid_df["img_path"], valid_df["male"].to_numpy().reshape(-1, 1), valid_df["bone_age_z"], use_gender=False, batch_size=batch_size, img_size=img_size)
        test_dataset = create_dataset_from_file(test_df["img_path"], test_df["male"].to_numpy().reshape(-1, 1), test_df["bone_age_z"], use_gender=False, batch_size=batch_size, img_size=img_size)
    else:
        train_dataset_wg = create_dataset_from_file(train_df["img_path"], train_df["male"].to_numpy().reshape(-1, 1), train_df["bone_age_z"], use_gender=True, batch_size=batch_size, img_size=img_size)
        valid_dataset_wg = create_dataset_from_file(valid_df["img_path"], valid_df["male"].to_numpy().reshape(-1, 1), valid_df["bone_age_z"], use_gender=True, batch_size=batch_size, img_size=img_size)
        test_dataset_wg = create_dataset_from_file(test_df["img_path"], test_df["male"].to_numpy().reshape(-1, 1), test_df["bone_age_z"], use_gender=True, batch_size=batch_size, img_size=img_size)

    # Weights and Biases run initialization
    run = wandb.init(project="hda-final", 
                    entity="hda-project",  # Entity is my team name on wandb website
                    # DA means data augmentation
                    name = f"inv4-DA-v1-g-{with_gender}",
                    config = {
                    "MODEL_NAME": "Inceptionv4",
                    "START_LR": lr,
                    "BATCH_SIZE": batch_size,
                    "GENDER": with_gender
                    })

    callbacks = [red_lr_plat, WandbCallback()]

    optimizer = tf.keras.optimizers.Adam( lr )

    if not with_gender:
        model = Inception((img_size, img_size, 3))()

        #compile model
        model.compile(loss = 'mse', optimizer = optimizer, metrics = [mae_in_months])

        # Train the model
        model.fit(train_dataset,  epochs = epoch, callbacks=callbacks, validation_data=valid_dataset)

        # predictions on test dataset
        test_mae = tf_dataset_calculate_mae_in_months(test_dataset)
    else:
        input_gender = tf.keras.Input(shape=(1), name="gender")
        model = Inception((img_size, img_size, 3), input_gender=input_gender)()

        #compile model
        model.compile(loss = 'mse', optimizer = optimizer, metrics = [mae_in_months])

        # Train the model
        model.fit(train_dataset_wg,  epochs = epoch, callbacks=callbacks, validation_data=valid_dataset_wg)

        # predictions on test dataset
        test_mae = tf_dataset_calculate_mae_in_months(test_dataset_wg)

    # # wandb automatically saves the model
    # art = wandb.Artifact(f"model-{run.name}-h5", type="model")
    # art.add_file(f"{run.dir}/model-best.h5")
    # wandb.log_artifact(art)

    # Log Performance of the test dataset
    wandb.log({"test_mae_in_months": test_mae})

    # Tell W&B that a model run is complete
    run.finish() 



