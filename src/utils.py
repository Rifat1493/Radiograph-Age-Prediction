import os
import random

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from wandb.keras import WandbCallback

import hparams
import wandb


def load_image(img_name):

    if isinstance(img_name, bytes):
        img_name = img_name.decode()

    img = cv2.imread(img_name, cv2.IMREAD_COLOR)
    img = np.array(cv2.resize(img, (IMG_SIZE, IMG_SIZE)), dtype="float32")

    return img


def normalize_img(image):
    return tf.cast(image, tf.float32) / 255.0


def set_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def create_dataset(file_names, labels, batch_size, shuffle, cache_file=None):
    # Create a Dataset object
    train_image_data = tf.data.Dataset.from_tensor_slices((file_names))

    # Map the load_image function
    py_func = lambda file_name: (tf.numpy_function(load_image, [file_name], tf.float32))
    train_image_data = train_image_data.map(py_func, num_parallel_calls=os.cpu_count())

    # Map the normalize_img function
    train_image_data = train_image_data.map(
        normalize_img, num_parallel_calls=os.cpu_count()
    )

    # Duplicate data for the autoencoder (input = output)
    # py_funct = lambda img: (img, img)
    # dataset = dataset.map(py_funct)
    labels = tf.convert_to_tensor(labels, dtype=tf.float32)
    train_label_data = tf.data.Dataset.from_tensor_slices((labels))

    dataset = tf.data.Dataset.zip((train_image_data, train_label_data))

    # Cache dataset
    if cache_file:
        dataset = dataset.cache(cache_file)

    # Shuffle
    if shuffle:
        dataset = dataset.shuffle(len(file_names))

    # Repeat the dataset indefinitely
    dataset = dataset.repeat()

    # Batch
    dataset = dataset.batch(batch_size=batch_size)

    # Prefetch
    dataset = dataset.prefetch(buffer_size=1)

    return dataset


def train_model(
    model,
    train_dataset=None,
    val_dataset=None,
    train_steps=None,
    val_steps=None,
    train_x=None,
    train_y=None,
    val_x=None,
    val_y=None,
    gend_train_x=None,
    gend_val_x=None
):
    # if hparams.GENDER:
    #     model_name = hparams.MODEL_NAME + "_gender"
    # else:
    #     model_name = hparams.MODEL_NAME

    if hparams.INIT_WB:
        wandb.init(
            project=hparams.PROJECT_NAME,
            entity="hda-project",
            name=hparams.MODEL_NAME
            # notes=hparams.NOTES
        )
        wandb.config.update(hparams.CONFIG)

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
        "../data/artifact/" + hparams.MODEL_NAME + ".h5",
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

    else:
        callbacks = [early_stopping, mc, red_lr_plat]

    # fit model

    if hparams.GENDER:
        history = model.fit(
            x=[train_x, gend_train_x],
            y=train_y,
            batch_size=hparams.BATCH_SIZE,
            epochs=hparams.EPOCHS,
            callbacks=callbacks,
            validation_data=([val_x, gend_val_x], val_y),
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            use_multiprocessing=True
        )

    else:

        history = model.fit_generator(
            train_dataset,
            steps_per_epoch=train_steps,
            validation_data=val_dataset,
            validation_steps=val_steps,
            epochs=hparams.EPOCHS,
            callbacks=callbacks,
        )

    return history
