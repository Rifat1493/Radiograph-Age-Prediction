from typing import Tuple

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.layers import *

import hparams
from layers2D import *


class BaselineCnn:
    @staticmethod
    def baseline_cnn():

        n_filters = 16
        batchnorm = hparams.BATCH_NORM

        inputs = Input(shape=(hparams.IMG_SIZE, hparams.IMG_SIZE, 3), name="input_1")

        c0 = conv2d_block(
            inputs,
            n_filters=n_filters,
            batchnorm=batchnorm,
            strides=1,
            recurrent=hparams.RECURRENT,
        )
        c1 = conv2d_block(
            c0,
            n_filters=n_filters * 2,
            batchnorm=batchnorm,
            strides=2,
            recurrent=hparams.RECURRENT,
        )
        c2 = conv2d_block(
            c1,
            n_filters=n_filters * 4,
            batchnorm=batchnorm,
            strides=2,
            recurrent=hparams.RECURRENT,
        )
        c3 = conv2d_block(
            c2,
            n_filters=n_filters * 8,
            batchnorm=batchnorm,
            strides=2,
            recurrent=hparams.RECURRENT,
        )
        c4 = conv2d_block(
            c3,
            n_filters=n_filters * 16,
            batchnorm=batchnorm,
            strides=2,
            recurrent=hparams.RECURRENT,
        )

        d1 = Flatten()(c4)
        d2 = Dense(16, activation=hparams.HID_ACT_FUNC)(d1)

        if hparams.GENDER:
            gd_input = Input(shape=(1,), name="input_2")
            gd_d1 = Dense(16, activation=hparams.HID_ACT_FUNC)(gd_input)
            d1_d2 = concatenate([d2, gd_d1])
            out = Dense(1, activation="linear", name="output")(d1_d2)
            model = Model(inputs=[inputs, gd_input], outputs=[out])
        else:
            out = Dense(1, activation="linear", name="output")(d2)
            model = Model(inputs=[inputs], outputs=[out])

        return model


class BaselineCnnAttention:
    @staticmethod
    def baseline_cnn_attention(sub_model_no):
        n_filters = 16
        batchnorm = hparams.BATCH_NORM

        inputs = Input(shape=(hparams.IMG_SIZE, hparams.IMG_SIZE, 3))

        #  1= one_attention_output_attention, 2= one_attention_output_cnn
        #  3= all_attention_output_attention, 4 = all_attention_output_cnn
        if sub_model_no == 1:

            c0 = conv2d_block(
                inputs,
                n_filters=n_filters,
                batchnorm=batchnorm,
                strides=1,
                recurrent=hparams.RECURRENT,
            )
            c1 = conv2d_block(
                c0,
                n_filters=n_filters * 2,
                batchnorm=batchnorm,
                strides=2,
                recurrent=hparams.RECURRENT,
            )
            c2 = conv2d_block(
                c1,
                n_filters=n_filters * 4,
                batchnorm=batchnorm,
                strides=2,
                recurrent=hparams.RECURRENT,
            )
            c3 = conv2d_block(
                c2,
                n_filters=n_filters * 8,
                batchnorm=batchnorm,
                strides=2,
                recurrent=hparams.RECURRENT,
            )

            c4 = conv2d_block(
                c3,
                n_filters=n_filters * 16,
                batchnorm=batchnorm,
                strides=2,
                recurrent=hparams.RECURRENT,
            )
            a0 = attn_gate_block(c3, c4, n_filters * 16)
            s_c4 = a0

        elif sub_model_no == 2:
            c0 = conv2d_block(
                inputs,
                n_filters=n_filters,
                batchnorm=batchnorm,
                strides=1,
                recurrent=hparams.RECURRENT,
            )
            c1 = conv2d_block(
                c0,
                n_filters=n_filters * 2,
                batchnorm=batchnorm,
                strides=2,
                recurrent=hparams.RECURRENT,
            )
            c2 = conv2d_block(
                c1,
                n_filters=n_filters * 4,
                batchnorm=batchnorm,
                strides=2,
                recurrent=hparams.RECURRENT,
            )
            c3 = conv2d_block(
                c2,
                n_filters=n_filters * 8,
                batchnorm=batchnorm,
                strides=2,
                recurrent=hparams.RECURRENT,
            )
            a0 = attn_gate_block(c2, c3, n_filters * 8)
            c4 = conv2d_block(
                a0,
                n_filters=n_filters * 16,
                batchnorm=batchnorm,
                strides=2,
                recurrent=hparams.RECURRENT,
            )
            s_c4 = c4

        elif sub_model_no == 3:
            c0 = conv2d_block(
                inputs,
                n_filters=n_filters,
                batchnorm=batchnorm,
                strides=1,
                recurrent=hparams.RECURRENT,
            )
            c1 = conv2d_block(
                c0,
                n_filters=n_filters * 2,
                batchnorm=batchnorm,
                strides=2,
                recurrent=hparams.RECURRENT,
            )

            a0 = attn_gate_block(c0, c1, n_filters * 2)
            c2 = conv2d_block(
                c1,
                n_filters=n_filters * 4,
                batchnorm=batchnorm,
                strides=2,
                recurrent=hparams.RECURRENT,
            )

            a1 = attn_gate_block(a0, c2, n_filters * 4)

            c3 = conv2d_block(
                c2,
                n_filters=n_filters * 8,
                batchnorm=batchnorm,
                strides=2,
                recurrent=hparams.RECURRENT,
            )

            a2 = attn_gate_block(a1, c3, n_filters * 8)
            c4 = conv2d_block(
                c3,
                n_filters=n_filters * 16,
                batchnorm=batchnorm,
                strides=2,
                recurrent=hparams.RECURRENT,
            )
            a3 = attn_gate_block(a2, c4, n_filters * 16)
            s_c4 = a3

        elif sub_model_no == 4:
            c0 = conv2d_block(
                inputs,
                n_filters=n_filters,
                batchnorm=batchnorm,
                strides=1,
                recurrent=hparams.RECURRENT,
            )
            c1 = conv2d_block(
                c0,
                n_filters=n_filters * 2,
                batchnorm=batchnorm,
                strides=2,
                recurrent=hparams.RECURRENT,
            )

            a0 = attn_gate_block(c0, c1, n_filters * 2)
            c2 = conv2d_block(
                c1,
                n_filters=n_filters * 4,
                batchnorm=batchnorm,
                strides=2,
                recurrent=hparams.RECURRENT,
            )

            a1 = attn_gate_block(a0, c2, n_filters * 4)

            c3 = conv2d_block(
                c2,
                n_filters=n_filters * 8,
                batchnorm=batchnorm,
                strides=2,
                recurrent=hparams.RECURRENT,
            )

            a2 = attn_gate_block(a1, c3, n_filters * 8)
            c4 = conv2d_block(
                a2,
                n_filters=n_filters * 8,
                batchnorm=batchnorm,
                strides=2,
                recurrent=hparams.RECURRENT,
            )

            s_c4 = c4

        d1 = Flatten()(s_c4)
        d2 = Dense(16, activation=hparams.HID_ACT_FUNC)(d1)

        if hparams.GENDER:
            gd_input = Input(shape=(1,))
            gd_d1 = Dense(16, activation=hparams.HID_ACT_FUNC)(gd_input)
            d1_d2 = concatenate([d2, gd_d1])

            out = Dense(1, activation="linear")(d1_d2)
            model = Model(inputs=[inputs, gd_input], outputs=[out])
        else:
            out = Dense(1, activation="linear")(d2)
            model = Model(inputs=[inputs], outputs=[out])

        return model


class SmallCNN:
    """
    Creates a small CNN:
        - without Gender branch
        - with Gender branch
    """
    def __init__(self, input_img, input_gender=None) -> None:

        self._model = None

        # CNN - For images
        x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="same", activation=None)(input_img)
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)

        x = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding="same", activation=None)(x)
        x = tf.keras.layers.BatchNormalization(axis=-1)(x)
        x = tf.keras.layers.Activation('relu')(x)

        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(x)

        x = tf.keras.layers.Flatten()(x)

        if input_gender is not None:
            # Dense - For Gender
            m = tf.keras.layers.Dense(32, activation='relu')(input_gender)

            # Concatenate CNN and Gender-Dense
            xm = tf.keras.layers.concatenate([x, m], name="concat-layer")

            xm = tf.keras.layers.Dense(64, activation='relu')(xm)
            xm = tf.keras.layers.Dense(32, activation='relu')(xm)
            xm = tf.keras.layers.Dense(16, activation='relu')(xm)
            age_output = tf.keras.layers.Dense(1, activation = 'linear')(xm)
            self._model = model = tf.keras.Model(inputs=[input_img, input_gender], outputs=age_output, name="Age_prediction_model")
        else:
            x = tf.keras.layers.Dense(64, activation='relu')(x)
            x = tf.keras.layers.Dense(32, activation='relu')(x)
            x = tf.keras.layers.Dense(16, activation='relu')(x)
            age_output = tf.keras.layers.Dense(1, activation = 'linear')(x)
            self._model = model = tf.keras.Model(inputs=input_img, outputs=age_output, name="Age_prediction_model")

    def __call__(self):
        """
        Returns the model created in __init__
        """
        return self._model


class Inception:
    def __init__(self, input_shape: Tuple[int]):
        """
        Implementation of the Inception-v4 architecture

        Arguments:
        input_shape -- tuple with dimension sizes of the images of the dataset e.g. (batch_size, n_H_prev, n_W_prev, n_channels)
        classes -- integer, number of classes

        Returns:
        model -- a Model() instance in Keras
        """

        # Define the input as a tensor with shape input_shape (1 line)
        X_input = Input(input_shape, name="img")

        # Call the above functions for the stem, inception-a, reduction-a, inception-b, reduction-b and inception-c blocks
        X = self._stem_block(X_input)

        # Four Inception A blocks
        X = self._inception_a_block(X, "a1")
        X = self._inception_a_block(X, "a2")
        X = self._inception_a_block(X, "a3")
        X = self._inception_a_block(X, "a4")

        # Reduction A block
        X = self._reduction_a_block(X)

        # Seven Inception B blocks
        X = self._inception_b_block(X, "b1")
        X = self._inception_b_block(X, "b2")
        X = self._inception_b_block(X, "b3")
        X = self._inception_b_block(X, "b4")
        X = self._inception_b_block(X, "b5")
        X = self._inception_b_block(X, "b6")
        X = self._inception_b_block(X, "b7")

        # Reduction B block
        X = self._reduction_b_block(X)

        # Three Inception C blocks
        X = self._inception_c_block(X, "c1")
        X = self._inception_c_block(X, "c2")
        X = self._inception_c_block(X, "c3")

        # AVGPOOL (1 line). Use "X = AveragePooling2D(...)(X)"
        kernel_pooling = X.get_shape()[1:3]
        X = AveragePooling2D(kernel_pooling, name="avg_pool")(X)
        X = Flatten()(X)

        # Dropout
        X = Dropout(rate=0.2)(X)

        # Output layer
        X = Dense(1, activation='linear', name='fc')(X)
        
        # Create model
        model = Model(inputs=X_input, outputs=X, name="Inceptionv4")

        # To be returned by __call__
        self._model = model

    def __call__(self):
        """
        Returns the model created in __init__
        """
        return self._model

    def _conv2d_bn(
        self,
        X_input,
        filters,
        kernel_size,
        strides,
        padding="same",
        activation=None,
        name: str = None,
    ):
        """
        Implementation of a conv block

        Arguments:
        X_input -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        filters -- integer, defining the number of filters in the CONV layer
        kernel_size -- (f1, f2) tuple of integers, specifying the shape of the CONV kernel
        s -- integer, specifying the stride to be used
        padding -- padding approach to be used
        name -- name for the layers

        Returns:
        X -- output of the conv2d_bn block, tensor of shape (n_H, n_W, n_C)
        """

        # defining name basis
        conv_name_base = "conv_"
        bn_name_base = "bn_"

        X = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            name=conv_name_base + name,
            kernel_initializer=glorot_uniform(seed=0),
        )(X_input)
        X = BatchNormalization(axis=3, name=bn_name_base + name)(X)
        if activation is not None:
            X = Activation(activation)(X)
        return X

    def _stem_block(self, X_input):
        """
        Implementation of the stem block as defined above

        Arguments:
        X_input -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)

        Returns:
        X -- output of the stem block, tensor of shape (n_H, n_W, n_C)

        branch1 - upper or left branch
        branch2 - lower or right branch
        """

        # First conv - inp:(299 x 299 x 3) -> out:(149 x 149 x 32)
        X = self._conv2d_bn(
            X_input=X_input,
            filters=32,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="valid",
            name="stem01",
        )

        # Second conv - inp:(149 x 149 x 32) -> out:(147 x 147 x 32)
        X = self._conv2d_bn(
            X_input=X,
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            name="stem02",
        )

        # Third conv - inp:(147 x 147 x 32) -> out:(147 x 147 x 64)
        X = self._conv2d_bn(
            X_input=X,
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            name="stem03",
        )

        # First branch: max pooling
        max_pool_s1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid")
        branch1 = max_pool_s1(X)

        # Second branch: conv
        branch2 = self._conv2d_bn(
            X_input=X,
            filters=96,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="valid",
            name="stem04",
        )

        # Concatenate (1) branch1 and branch2 along the channel axis
        X = tf.concat([branch1, branch2], axis=3, name="concat-01")

        # First branch: 2 convs
        branch1 = self._conv2d_bn(
            X_input=X,
            filters=64,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            name="stem-05",
        )

        branch1 = self._conv2d_bn(
            X_input=branch1,
            filters=96,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            name="stem-06",
        )

        # Second branch: 4 convs
        branch2 = self._conv2d_bn(
            X_input=X,
            filters=64,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            name="stem-07",
        )
        branch2 = self._conv2d_bn(
            X_input=branch2,
            filters=64,
            kernel_size=(7, 1),
            strides=(1, 1),
            padding="same",
            name="stem-08",
        )
        branch2 = self._conv2d_bn(
            X_input=branch2,
            filters=64,
            kernel_size=(1, 7),
            strides=(1, 1),
            padding="same",
            name="stem-09",
        )
        branch2 = self._conv2d_bn(
            X_input=branch2,
            filters=96,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="valid",
            name="stem-10",
        )

        # Concatenate (2) branch1 and branch2 along the channel axis
        X = tf.concat([branch1, branch2], axis=3, name="concat-02")

        # First branch: conv
        branch1 = self._conv2d_bn(
            X_input=X,
            filters=192,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="valid",
            name="stem-11",
        )

        # Second branch: max pooling
        branch2 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding="valid")(X)

        # Concatenate (3) branch1 and branch2 along the channel axis
        X = tf.concat([branch1, branch2], axis=-1, name="concat-03")

        return X

    def _inception_a_block(self, X_input, base_name):
        """
        Implementation of the Inception-A block

        Arguments:
        X_input -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)

        Returns:
        X -- output of the block, tensor of shape (n_H, n_W, n_C)
        """

        # Branch 1
        branch1 = AveragePooling2D(
            pool_size=(3, 3),
            strides=(1, 1),
            padding="same",
            name=base_name + "ia_branch_1_1",
        )(X_input)
        branch1 = self._conv2d_bn(
            branch1,
            filters=96,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name=base_name + "ia_branch_1_2",
        )

        # Branch 2
        branch2 = self._conv2d_bn(
            X_input,
            filters=96,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name=base_name + "ia_branch_2_1",
        )

        # Branch 3
        branch3 = self._conv2d_bn(
            X_input,
            filters=64,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name=base_name + "ia_branch_3_1",
        )
        branch3 = self._conv2d_bn(
            branch3,
            filters=96,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name=base_name + "ia_branch_3_2",
        )

        # Branch 4
        branch4 = self._conv2d_bn(
            X_input,
            filters=64,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name=base_name + "ia_branch_4_1",
        )
        branch4 = self._conv2d_bn(
            branch4,
            filters=96,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name=base_name + "ia_branch_4_2",
        )
        branch4 = self._conv2d_bn(
            branch4,
            filters=96,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name=base_name + "ia_branch_4_3",
        )

        # Concatenate branch1, branch2, branch3 and branch4 along the channel axis
        X = tf.concat(values=[branch1, branch2, branch3, branch4], axis=3)

        return X

    def _inception_b_block(self, X_input, base_name):
        """
        Implementation of the Inception-B block

        Arguments:
        X_input -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)

        Returns:
        X -- output of the block, tensor of shape (n_H, n_W, n_C)
        """

        # Branch 1
        branch1 = AveragePooling2D(
            pool_size=(3, 3),
            strides=(1, 1),
            padding="same",
            name=base_name + "ib_branch_1_1",
        )(X_input)
        branch1 = self._conv2d_bn(
            branch1,
            filters=128,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name=base_name + "ib_branch_1_2",
        )

        # Branch 2
        branch2 = self._conv2d_bn(
            X_input,
            filters=384,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name=base_name + "ib_branch_2_1",
        )

        # Branch 3
        branch3 = self._conv2d_bn(
            X_input,
            filters=192,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name=base_name + "ib_branch_3_1",
        )
        branch3 = self._conv2d_bn(
            branch3,
            filters=224,
            kernel_size=(1, 7),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name=base_name + "ib_branch_3_2",
        )
        branch3 = self._conv2d_bn(
            branch3,
            filters=256,
            kernel_size=(7, 1),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name=base_name + "ib_branch_3_3",
        )

        # Branch 4
        branch4 = self._conv2d_bn(
            X_input,
            filters=192,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name=base_name + "ib_branch_4_1",
        )
        branch4 = self._conv2d_bn(
            branch4,
            filters=192,
            kernel_size=(1, 7),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name=base_name + "ib_branch_4_2",
        )
        branch4 = self._conv2d_bn(
            branch4,
            filters=224,
            kernel_size=(7, 1),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name=base_name + "ib_branch_4_3",
        )
        branch4 = self._conv2d_bn(
            branch4,
            filters=224,
            kernel_size=(1, 7),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name=base_name + "ib_branch_4_4",
        )
        branch4 = self._conv2d_bn(
            branch4,
            filters=256,
            kernel_size=(7, 1),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name=base_name + "ib_branch_4_5",
        )

        # Concatenate branch1, branch2, branch3 and branch4 along the channel axis
        X = tf.concat(values=[branch1, branch2, branch3, branch4], axis=3)

        ### END CODE HERE ###

        return X

    def _inception_c_block(self, X_input, base_name):
        """
        Implementation of the Inception-C block. Branches go from left to right

        Arguments:
        X_input -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)

        Returns:
        X -- output of the block, tensor of shape (n_H, n_W, n_C)
        """

        # Branch 1
        branch1 = AveragePooling2D(
            pool_size=(3, 3),
            strides=(1, 1),
            padding="same",
            name=base_name + "ic_branch_1_1",
        )(X_input)
        branch1 = self._conv2d_bn(
            branch1,
            filters=256,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name=base_name + "ic_branch_1_2",
        )

        # Branch 2
        branch2 = self._conv2d_bn(
            X_input,
            filters=256,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name=base_name + "ic_branch_2_1",
        )

        # Branch 3
        branch3 = self._conv2d_bn(
            X_input,
            filters=384,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name=base_name + "ic_branch_3_1",
        )
        branch3_1 = self._conv2d_bn(
            branch3,
            filters=256,
            kernel_size=(1, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name=base_name + "ic_branch_3_2",
        )
        branch3_2 = self._conv2d_bn(
            branch3,
            filters=256,
            kernel_size=(3, 1),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name=base_name + "ic_branch_3_3",
        )

        # Branch 4
        branch4 = self._conv2d_bn(
            X_input,
            filters=384,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name=base_name + "ic_branch_4_1",
        )
        branch4 = self._conv2d_bn(
            branch4,
            filters=448,
            kernel_size=(1, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name=base_name + "ic_branch_4_2",
        )
        branch4 = self._conv2d_bn(
            branch4,
            filters=512,
            kernel_size=(3, 1),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name=base_name + "ic_branch_4_3",
        )
        branch4_1 = self._conv2d_bn(
            branch4,
            filters=256,
            kernel_size=(3, 1),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name=base_name + "ic_branch_4_4",
        )
        branch4_2 = self._conv2d_bn(
            branch4,
            filters=256,
            kernel_size=(1, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name=base_name + "ic_branch_4_5",
        )

        # Concatenate branch1, branch2, branch3_1, branch3_2, branch4_1 and branch4_2 along the channel axis
        X = tf.concat(
            values=[branch1, branch2, branch3_1, branch3_2, branch4_1, branch4_2],
            axis=3,
        )

        return X

    def _reduction_a_block(self, X_input):
        """
        Implementation of the Reduction-A block

        Arguments:
        X_input -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)

        Returns:
        X -- output of the block, tensor of shape (n_H, n_W, n_C)
        """

        # Branch 1: Max Pool
        branch1 = MaxPool2D(
            pool_size=(3, 3), strides=(2, 2), padding="valid", name="ra_branch_1_1"
        )(X_input)

        # Branch 2
        branch2 = self._conv2d_bn(
            X_input,
            filters=384,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="valid",
            activation="relu",
            name="ra_branch_2_1",
        )

        # Branch 3
        branch3 = self._conv2d_bn(
            X_input,
            filters=192,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name="ra_branch_3_1",
        )
        branch3 = self._conv2d_bn(
            branch3,
            filters=224,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name="ra_branch_3_2",
        )
        branch3 = self._conv2d_bn(
            branch3,
            filters=256,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="valid",
            activation="relu",
            name="ra_branch_3_3",
        )

        # Concatenate branch1, branch2 and branch3 along the channel axis
        X = tf.concat(values=[branch1, branch2, branch3], axis=3)

        return X

    def _reduction_b_block(self, X_input):
        """
        Implementation of the Reduction-B block

        Arguments:
        X_input -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)

        Returns:
        X -- output of the block, tensor of shape (n_H, n_W, n_C)
        """

        # Branch 1
        branch1 = MaxPool2D(
            pool_size=(3, 3), strides=(2, 2), padding="valid", name="rb_branch_1_1"
        )(X_input)

        # Branch 2
        branch2 = self._conv2d_bn(
            X_input,
            filters=192,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name="rb_branch_2_1",
        )
        branch2 = self._conv2d_bn(
            branch2,
            filters=192,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="valid",
            activation="relu",
            name="rb_branch_2_2",
        )

        # Branch 3
        branch3 = self._conv2d_bn(
            X_input,
            filters=256,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name="rb_branch_3_1",
        )
        branch3 = self._conv2d_bn(
            branch3,
            filters=256,
            kernel_size=(1, 7),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name="rb_branch_3_2",
        )
        branch3 = self._conv2d_bn(
            branch3,
            filters=320,
            kernel_size=(7, 1),
            strides=(1, 1),
            padding="same",
            activation="relu",
            name="rb_branch_3_3",
        )
        branch3 = self._conv2d_bn(
            branch3,
            filters=320,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="valid",
            activation="relu",
            name="rb_branch_3_4",
        )

        # Concatenate branch1, branch2 and branch3 along the channel axis
        X = tf.concat(values=[branch1, branch2, branch3], axis=3)

        return X

    

class ResidualAttentionUnet:
    @staticmethod
    def residual_attention_unet():

        n_filters = 16
        batchnorm = hparams.BATCH_NORM

        inputs = Input(shape=(hparams.IMG_SIZE, hparams.IMG_SIZE, 3))

        c0 = residual_block(
            inputs,
            n_filters=n_filters,
            batchnorm=batchnorm,
            strides=1,
            recurrent=hparams.RECURRENT,
        )  # 512x512x512

        c1 = residual_block(
            c0,
            n_filters=n_filters * 2,
            batchnorm=batchnorm,
            strides=2,
            recurrent=hparams.RECURRENT,
        )  # 256x256x256

        c2 = residual_block(
            c1,
            n_filters=n_filters * 4,
            batchnorm=batchnorm,
            strides=2,
            recurrent=hparams.RECURRENT,
        )  # 128x128x128

        c3 = residual_block(
            c2,
            n_filters=n_filters * 8,
            batchnorm=batchnorm,
            strides=2,
            recurrent=hparams.RECURRENT,
        )  # 64x64x64

        # bridge

        b0 = residual_block(
            c3,
            n_filters=n_filters * 16,
            batchnorm=batchnorm,
            strides=2,
            recurrent=hparams.RECURRENT,
        )  # 32x32x32

        # expansive path

        attn0 = attn_gate_block(c3, b0, n_filters * 16)
        u0 = transpose_block(
            b0,
            attn0,
            n_filters=n_filters * 8,
            batchnorm=batchnorm,
            recurrent=hparams.RECURRENT,
        )  # 64x64x64

        attn1 = attn_gate_block(c2, u0, n_filters * 8)
        u1 = transpose_block(
            u0,
            attn1,
            n_filters=n_filters * 4,
            batchnorm=batchnorm,
            recurrent=hparams.RECURRENT,
        )  # 128x128x128

        attn2 = attn_gate_block(c1, u1, n_filters * 4)
        u2 = transpose_block(
            u1,
            attn2,
            n_filters=n_filters * 2,
            batchnorm=batchnorm,
            recurrent=hparams.RECURRENT,
        )  # 256x256x256

        u3 = transpose_block(
            u2,
            c0,
            n_filters=n_filters,
            batchnorm=batchnorm,
            recurrent=hparams.RECURRENT,
        )  # 512x512x512

        c9 = Conv2D(
            16,
            (3, 3),
            activation=hparams.HID_ACT_FUNC,
            kernel_initializer="he_normal",
            padding="same",
        )(u3)
        d1 = Flatten()(c9)
        d2 = Dense(16, activation=hparams.HID_ACT_FUNC)(d1)
        if hparams.GENDER:
            gd_input = Input(shape=(1,))
            gd_d1 = Dense(16, activation=hparams.HID_ACT_FUNC)(gd_input)
            d1_d2 = concatenate([d2, gd_d1])

            out = Dense(1, activation="linear")(d1_d2)
            model = Model(inputs=[inputs, gd_input], outputs=[out])
        else:
            out = Dense(1, activation="linear")(d2)
            model = Model(inputs=[inputs], outputs=[out])
        return model


class InceptionAttentionUnet:
    @staticmethod
    def inception_attention_unet():
        n_filters = 16
        batchnorm = hparams.BATCH_NORM

        inputs = Input(shape=(hparams.IMG_SIZE, hparams.IMG_SIZE, 3))

        c0 = inception_block(
            inputs,
            n_filters=n_filters,
            batchnorm=batchnorm,
            strides=1,
            recurrent=hparams.RECURRENT,
            layers=[[(3, 1), (3, 1)], [(3, 2)]],
        )  # 512x512x512

        c1 = inception_block(
            c0,
            n_filters=n_filters * 2,
            batchnorm=batchnorm,
            strides=2,
            recurrent=hparams.RECURRENT,
            layers=[[(3, 1), (3, 1)], [(3, 2)]],
        )  # 256x256x256

        c2 = inception_block(
            c1,
            n_filters=n_filters * 4,
            batchnorm=batchnorm,
            strides=2,
            recurrent=hparams.RECURRENT,
            layers=[[(3, 1), (3, 1)], [(3, 2)]],
        )  # 128x128x128

        c3 = inception_block(
            c2,
            n_filters=n_filters * 8,
            batchnorm=batchnorm,
            strides=2,
            recurrent=hparams.RECURRENT,
            layers=[[(3, 1), (3, 1)], [(3, 2)]],
        )  # 64x64x64

        # bridge

        b0 = inception_block(
            c3,
            n_filters=n_filters * 16,
            batchnorm=batchnorm,
            strides=2,
            recurrent=hparams.RECURRENT,
            layers=[[(3, 1), (3, 1)], [(3, 2)]],
        )  # 32x32x32

        # expansive path

        attn0 = attn_gate_block(c3, b0, n_filters * 16)
        u0 = transpose_block(
            b0,
            attn0,
            n_filters=n_filters * 8,
            batchnorm=batchnorm,
            recurrent=hparams.RECURRENT,
        )  # 64x64x64

        attn1 = attn_gate_block(c2, u0, n_filters * 8)
        u1 = transpose_block(
            u0,
            attn1,
            n_filters=n_filters * 4,
            batchnorm=batchnorm,
            recurrent=hparams.RECURRENT,
        )  # 128x128x128

        attn2 = attn_gate_block(c1, u1, n_filters * 4)
        u2 = transpose_block(
            u1,
            attn2,
            n_filters=n_filters * 2,
            batchnorm=batchnorm,
            recurrent=hparams.RECURRENT,
        )  # 256x256x256

        u3 = transpose_block(
            u2,
            c0,
            n_filters=n_filters,
            batchnorm=batchnorm,
            recurrent=hparams.RECURRENT,
        )  # 512x512x512

        c9 = Conv2D(
            16,
            (3, 3),
            activation=hparams.HID_ACT_FUNC,
            kernel_initializer="he_normal",
            padding="same",
        )(u3)
        d1 = Flatten()(c9)
        d2 = Dense(16, activation=hparams.HID_ACT_FUNC)(d1)
        if hparams.GENDER:
            gd_input = Input(shape=(1,))
            gd_d1 = Dense(16, activation=hparams.HID_ACT_FUNC)(gd_input)
            d1_d2 = concatenate([d2, gd_d1])

            out = Dense(1, activation="linear")(d1_d2)
            model = Model(inputs=[inputs, gd_input], outputs=[out])
        else:
            out = Dense(1, activation="linear")(d2)
            model = Model(inputs=[inputs], outputs=[out])
        return model


class CnnAttentionUnet:
    @staticmethod
    def cnn_attention_unet():
        n_filters = 16
        batchnorm = hparams.BATCH_NORM

        inputs = Input(shape=(hparams.IMG_SIZE, hparams.IMG_SIZE, 3))

        c0 = conv2d_block(
            inputs,
            n_filters=n_filters,
            batchnorm=batchnorm,
            strides=1,
            recurrent=hparams.RECURRENT,
        )  # 512x512x512

        c1 = conv2d_block(
            c0,
            n_filters=n_filters * 2,
            batchnorm=batchnorm,
            strides=2,
            recurrent=hparams.RECURRENT,
        )  # 256x256x256

        c2 = conv2d_block(
            c1,
            n_filters=n_filters * 4,
            batchnorm=batchnorm,
            strides=2,
            recurrent=hparams.RECURRENT,
        )  # 128x128x128

        c3 = conv2d_block(
            c2,
            n_filters=n_filters * 8,
            batchnorm=batchnorm,
            strides=2,
            recurrent=hparams.RECURRENT,
        )  # 64x64x64

        # bridge

        b0 = conv2d_block(
            c3,
            n_filters=n_filters * 16,
            batchnorm=batchnorm,
            strides=2,
            recurrent=hparams.RECURRENT,
        )  # 32x32x32

        # expansive path

        attn0 = attn_gate_block(c3, b0, n_filters * 16)
        u0 = transpose_block(
            b0,
            attn0,
            n_filters=n_filters * 8,
            batchnorm=batchnorm,
            recurrent=hparams.RECURRENT,
        )  # 64x64x64

        attn1 = attn_gate_block(c2, u0, n_filters * 8)
        u1 = transpose_block(
            u0,
            attn1,
            n_filters=n_filters * 4,
            batchnorm=batchnorm,
            recurrent=hparams.RECURRENT,
        )  # 128x128x128

        attn2 = attn_gate_block(c1, u1, n_filters * 4)
        u2 = transpose_block(
            u1,
            attn2,
            n_filters=n_filters * 2,
            batchnorm=batchnorm,
            recurrent=hparams.RECURRENT,
        )  # 256x256x256

        u3 = transpose_block(
            u2,
            c0,
            n_filters=n_filters,
            batchnorm=batchnorm,
            recurrent=hparams.RECURRENT,
        )  # 512x512x512

        c9 = Conv2D(
            16,
            (3, 3),
            activation=hparams.HID_ACT_FUNC,
            kernel_initializer="he_normal",
            padding="same",
        )(u3)
        d1 = Flatten()(c9)
        d2 = Dense(16, activation=hparams.HID_ACT_FUNC)(d1)
        if hparams.GENDER:
            gd_input = Input(shape=(1,))
            gd_d1 = Dense(16, activation=hparams.HID_ACT_FUNC)(gd_input)
            d1_d2 = concatenate([d2, gd_d1])

            out = Dense(1, activation="linear")(d1_d2)
            model = Model(inputs=[inputs, gd_input], outputs=[out])
        else:
            out = Dense(1, activation="linear")(d2)
            model = Model(inputs=[inputs], outputs=[out])
        return model


class Unet:
    @staticmethod
    def unet():
        inputs = Input(shape=(hparams.IMG_SIZE, hparams.IMG_SIZE, 3))
        # Contraction path
        c1 = Conv2D(
            16,
            (3, 3),
            activation=hparams.HID_ACT_FUNC,
            kernel_initializer="he_normal",
            padding="same",
        )(inputs)
        c1 = BatchNormalization()(c1)
        c1 = Dropout(0.1)(c1)
        c1 = Conv2D(
            16,
            (3, 3),
            activation=hparams.HID_ACT_FUNC,
            kernel_initializer="he_normal",
            padding="same",
        )(c1)
        c1 = BatchNormalization()(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(
            32,
            (3, 3),
            activation=hparams.HID_ACT_FUNC,
            kernel_initializer="he_normal",
            padding="same",
        )(p1)
        c2 = BatchNormalization()(c2)
        c2 = Dropout(0.1)(c2)
        c2 = Conv2D(
            32,
            (3, 3),
            activation=hparams.HID_ACT_FUNC,
            kernel_initializer="he_normal",
            padding="same",
        )(c2)
        c2 = BatchNormalization()(c2)
        p2 = MaxPooling2D((2, 2))(c2)

        c3 = Conv2D(
            64,
            (3, 3),
            activation=hparams.HID_ACT_FUNC,
            kernel_initializer="he_normal",
            padding="same",
        )(p2)
        c3 = BatchNormalization()(c3)
        c3 = Dropout(0.2)(c3)
        c3 = Conv2D(
            64,
            (3, 3),
            activation=hparams.HID_ACT_FUNC,
            kernel_initializer="he_normal",
            padding="same",
        )(c3)

        ## add batch
        c3 = BatchNormalization()(c3)

        p3 = MaxPooling2D((2, 2))(c3)

        c4 = Conv2D(
            128,
            (3, 3),
            activation=hparams.HID_ACT_FUNC,
            kernel_initializer="he_normal",
            padding="same",
        )(p3)
        c4 = BatchNormalization()(c4)
        c4 = Dropout(0.2)(c4)
        c4 = Conv2D(
            128,
            (3, 3),
            activation=hparams.HID_ACT_FUNC,
            kernel_initializer="he_normal",
            padding="same",
        )(c4)
        c4 = BatchNormalization()(c4)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)

        c5 = Conv2D(
            256,
            (3, 3),
            activation=hparams.HID_ACT_FUNC,
            kernel_initializer="he_normal",
            padding="same",
        )(p4)
        c5 = BatchNormalization()(c5)
        c5 = Dropout(0.3)(c5)
        c5 = Conv2D(
            256,
            (3, 3),
            activation=hparams.HID_ACT_FUNC,
            kernel_initializer="he_normal",
            padding="same",
        )(c5)
        c5 = BatchNormalization()(c5)
        # Expansive path
        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(
            128,
            (3, 3),
            activation=hparams.HID_ACT_FUNC,
            kernel_initializer="he_normal",
            padding="same",
        )(u6)
        c6 = BatchNormalization()(c6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(
            128,
            (3, 3),
            activation=hparams.HID_ACT_FUNC,
            kernel_initializer="he_normal",
            padding="same",
        )(c6)
        c6 = BatchNormalization()(c6)
        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(
            64,
            (3, 3),
            activation=hparams.HID_ACT_FUNC,
            kernel_initializer="he_normal",
            padding="same",
        )(u7)
        c7 = BatchNormalization()(c7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv2D(
            64,
            (3, 3),
            activation=hparams.HID_ACT_FUNC,
            kernel_initializer="he_normal",
            padding="same",
        )(c7)
        c7 = BatchNormalization()(c7)
        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(
            32,
            (3, 3),
            activation=hparams.HID_ACT_FUNC,
            kernel_initializer="he_normal",
            padding="same",
        )(u8)
        c8 = BatchNormalization()(c8)
        c8 = Dropout(0.1)(c8)
        c8 = Conv2D(
            32,
            (3, 3),
            activation=hparams.HID_ACT_FUNC,
            kernel_initializer="he_normal",
            padding="same",
        )(c8)
        c8 = BatchNormalization()(c8)
        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(
            16,
            (3, 3),
            activation=hparams.HID_ACT_FUNC,
            kernel_initializer="he_normal",
            padding="same",
        )(u9)
        c9 = BatchNormalization()(c9)
        c9 = Dropout(0.1)(c9)
        c9 = Conv2D(
            16,
            (3, 3),
            activation=hparams.HID_ACT_FUNC,
            kernel_initializer="he_normal",
            padding="same",
        )(c9)
        c9 = BatchNormalization()(c9)
        d1 = Flatten()(c9)
        d2 = Dense(16, activation=hparams.HID_ACT_FUNC)(d1)
        if hparams.GENDER:
            gd_input = Input(shape=(1,))
            gd_d1 = Dense(16, activation=hparams.HID_ACT_FUNC)(gd_input)
            d1_d2 = concatenate([d2, gd_d1])

            out = Dense(1, activation="linear")(d1_d2)
            model = Model(inputs=[inputs, gd_input], outputs=[out])
        else:
            out = Dense(1, activation="linear")(d2)
            model = Model(inputs=[inputs], outputs=[out])

        return model
