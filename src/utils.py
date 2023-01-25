import os
import random

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from wandb.keras import WandbCallback

import hparams
import wandb


def load_image(img_name):

    if isinstance(img_name, bytes):
        img_name = img_name.decode()

    img = cv2.imread(img_name, cv2.IMREAD_COLOR)
    img = np.array(
        cv2.resize(img, (hparams.IMG_SIZE, hparams.IMG_SIZE)), dtype="float32"
    )

    return img


def map_fn(input_1, input_2, output):
    return ({"input_1": input_1, "input_2": input_2}, output)


def normalize_img(image):
    return tf.cast(image, tf.float32) / 255.0


def set_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def create_dataset_from_file(file_names, gend_array, y_array, use_gender=hparams.GENDER, batch_size = hparams.BATCH_SIZE):
    # Create a Dataset object
    train_image_data = tf.data.Dataset.from_tensor_slices((file_names))

    # Map the load_image function
    py_func = lambda file_name: (tf.numpy_function(load_image, [file_name], tf.float32))
    train_image_data = train_image_data.map(py_func, num_parallel_calls=os.cpu_count())

    # Map the normalize_img function
    generator_dataset = train_image_data.map(
        normalize_img, num_parallel_calls=os.cpu_count()
    )

    y_array = tf.convert_to_tensor(y_array, dtype=tf.float32)
    gend_array = tf.convert_to_tensor(gend_array, dtype=tf.float32)
    array_dataset = tf.data.Dataset.from_tensor_slices((gend_array))
    y_array_dataset = tf.data.Dataset.from_tensor_slices(y_array)

    # Zip the two dataset objects together
    # tmp_dataset = tf.data.Dataset.zip((generator_dataset, array_dataset))

    # dataset = tf.data.Dataset.zip((tmp_dataset, y_array_dataset))

    # dataset = tf.data.Dataset.zip((generator_dataset, array_dataset, y_array_dataset))
    if use_gender:

        dataset = tf.data.Dataset.zip(
            ((generator_dataset, array_dataset), y_array_dataset)
        )
    else:
        dataset = tf.data.Dataset.zip((generator_dataset, y_array_dataset))
    # dataset = dataset.map(map_fn)

    # # Cache dataset
    # if cache_file:
    #     dataset = dataset.cache(cache_file)

    # # Shuffle
    # if shuffle:
    #     dataset = dataset.shuffle(len(file_names))

    # # Repeat the dataset indefinitely
    # dataset = dataset.repeat()

    # # Batch
    dataset = dataset.batch(batch_size)

    # # Prefetch
    # dataset = dataset.prefetch(buffer_size=1)

    return dataset


def train_model(
    model,
    train_dataset=None,
    val_dataset=None,
    train_steps=None,
    val_steps=None,
):
    # if hparams.GENDER:
    #     model_name = hparams.MODEL_NAME + "_gender"
    # else:
    #     model_name = hparams.MODEL_NAME

    # config = hparams.CONFIG
    # config["project"] = hparams.PROJECT_NAME
    # config["entity"] = "hda-project"
    # config["name"] = hparams.MODEL_NAME

    # wandb.init(config=config)

    # wandb.init(
    #     project=hparams.PROJECT_NAME,
    #     entity="hda-project",
    #     name=hparams.MODEL_NAME
    #     # notes=hparams.NOTES
    # )
    # wandb.config.update(hparams.CONFIG,allow_val_change=True)

    # early stopping
    # patience=5
    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=hparams.PATIENCE,
        verbose=0,
        mode="min",
    )

    # model checkpoint
    mc = ModelCheckpoint(
        "data/artifact/" + hparams.MODEL_NAME + ".h5",
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )

    # tensorboard callback
    # logdir = os.path.join(logs_dir,datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    # tensorboard_callback =  TensorBoard(logdir, histogram_freq = 1)

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

    if hparams.INIT_WB:
        callbacks = [
            early_stopping,
            mc,
            red_lr_plat,
            WandbCallback(mode="min", save_model=False),
        ]
#red_lr_plat,
    else:
        callbacks = [early_stopping, mc, red_lr_plat]

    # fit model
    history = model.fit(
        train_dataset,
        steps_per_epoch=train_steps,
        validation_data=val_dataset,
        validation_steps=val_steps,
        epochs=hparams.EPOCHS,
        callbacks=callbacks,
    )

    return history


def make_gen_callable(_gen, array, y_array):
    def gen():
        for x, y, z in zip(_gen, array, y_array):
            inputs = (x, y)
            outputs = z
            yield inputs, outputs

    return gen
