import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPool2D, Flatten, Add, Activation, ZeroPadding2D, \
                                    BatchNormalization, AveragePooling2D, GlobalMaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
import hparams
from typing import Tuple
from layers2D import *


class BaselineCnn:
    @staticmethod
    def baseline_cnn():

        n_filters = 16
        batchnorm = BATCH_NORM

        inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

        c0 = conv2d_block(
            inputs, n_filters=n_filters, batchnorm=batchnorm, strides=1, recurrent=1
        )
        c1 = conv2d_block(
            c0, n_filters=n_filters * 2, batchnorm=batchnorm, strides=2, recurrent=1
        )
        c2 = conv2d_block(
            c1, n_filters=n_filters * 4, batchnorm=batchnorm, strides=2, recurrent=1
        )
        c3 = conv2d_block(
            c2, n_filters=n_filters * 8, batchnorm=batchnorm, strides=2, recurrent=1
        )
        c4 = conv2d_block(
            c3, n_filters=n_filters * 16, batchnorm=batchnorm, strides=2, recurrent=1
        )

        d1 = Flatten()(c4)
        d2 = Dense(16, activation="elu")(d1)
        out = Dense(1, activation="linear")(d2)
        model = Model(inputs=[inputs], outputs=[out])

        return model


class BaselineCnnAttention:
    @staticmethod
    def baseline_cnn_attention():
        n_filters = 16
        batchnorm = BATCH_NORM

        inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

        c0 = conv2d_block(
            inputs, n_filters=n_filters, batchnorm=batchnorm, strides=1, recurrent=2
        )
        c1 = conv2d_block(
            c0, n_filters=n_filters * 2, batchnorm=batchnorm, strides=2, recurrent=2
        )
        c2 = conv2d_block(
            c1, n_filters=n_filters * 4, batchnorm=batchnorm, strides=2, recurrent=2
        )
        c3 = conv2d_block(
            c2, n_filters=n_filters * 8, batchnorm=batchnorm, strides=2, recurrent=2
        )
        ao = attn_gate_block(c2, c3, n_filters * 8)
        c4 = conv2d_block(
            ao, n_filters=n_filters * 16, batchnorm=batchnorm, strides=2, recurrent=2
        )

        d1 = Flatten()(c4)
        d2 = Dense(16, activation="elu")(d1)
        out = Dense(1, activation="linear")(d2)
        model = Model(inputs=[inputs], outputs=[out])

        return model


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
        X = self._inception_a_block(X, 'a1')
        X = self._inception_a_block(X, 'a2')
        X = self._inception_a_block(X, 'a3')
        X = self._inception_a_block(X, 'a4')

        # Reduction A block
        X = self._reduction_a_block(X)

        # Seven Inception B blocks
        X = self._inception_b_block(X, 'b1')
        X = self._inception_b_block(X, 'b2')
        X = self._inception_b_block(X, 'b3')
        X = self._inception_b_block(X, 'b4')
        X = self._inception_b_block(X, 'b5')
        X = self._inception_b_block(X, 'b6')
        X = self._inception_b_block(X, 'b7')

        # Reduction B block
        X = self._reduction_b_block(X)

        # Three Inception C blocks
        X = self._inception_c_block(X, 'c1')
        X = self._inception_c_block(X, 'c2')
        X = self._inception_c_block(X, 'c3')

        # AVGPOOL (1 line). Use "X = AveragePooling2D(...)(X)"
        kernel_pooling = X.get_shape()[1:3]
        X = AveragePooling2D(kernel_pooling, name='avg_pool')(X)
        X = Flatten()(X)

        # Dropout
        X = Dropout(rate = 0.2)(X)

        # Output layer
        X = Dense(1, activation='sigmoid', name='fc')(X)
        
        # Create model
        model = Model(inputs = X_input, outputs = X, name='Inceptionv4')

        # To be returned by __call__
        self._model = model


    def __call__(self):
        """
        Returns the model created in __init__ 
        """
        return self._model        
        

    def _conv2d_bn(self, X_input, filters, kernel_size, strides, padding='same', activation=None, name: str = None):
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
        conv_name_base = 'conv_'
        bn_name_base = 'bn_'

        X = Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, 
                padding = padding, name = conv_name_base + name, 
                kernel_initializer = glorot_uniform(seed=0))(X_input)
        X = BatchNormalization(axis = 3, name = bn_name_base + name)(X)
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
        X = self._conv2d_bn(X_input=X_input, filters=32, kernel_size=(3, 3), strides=(2, 2), padding='valid', name='stem01')
        
        # Second conv - inp:(149 x 149 x 32) -> out:(147 x 147 x 32)
        X = self._conv2d_bn(X_input=X, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='stem02')

        # Third conv - inp:(147 x 147 x 32) -> out:(147 x 147 x 64)
        X = self._conv2d_bn(X_input=X, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='stem03')

        # First branch: max pooling
        max_pool_s1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')
        branch1 = max_pool_s1(X)

        # Second branch: conv
        branch2 = self._conv2d_bn(X_input=X, filters=96, kernel_size=(3, 3), strides=(2, 2), padding='valid', name='stem04')

        # Concatenate (1) branch1 and branch2 along the channel axis
        X = tf.concat([branch1, branch2], axis=3, name='concat-01')

        # First branch: 2 convs
        branch1 = self._conv2d_bn(X_input=X, filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', name='stem-05')
        
        branch1 = self._conv2d_bn(X_input=branch1, filters=96, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='stem-06')
        
        # Second branch: 4 convs
        branch2 = self._conv2d_bn(X_input=X, filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', name='stem-07')
        branch2 = self._conv2d_bn(X_input=branch2, filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', name='stem-08')
        branch2 = self._conv2d_bn(X_input=branch2, filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', name='stem-09')
        branch2 = self._conv2d_bn(X_input=branch2, filters=96, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='stem-10')

        # Concatenate (2) branch1 and branch2 along the channel axis
        X = tf.concat([branch1, branch2], axis=3, name='concat-02')

        # First branch: conv
        branch1 = self._conv2d_bn(X_input=X, filters=192, kernel_size=(3, 3), strides=(2, 2), padding='valid', name='stem-11')

        # Second branch: max pooling
        branch2 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(X)

        # Concatenate (3) branch1 and branch2 along the channel axis
        X = tf.concat([branch1, branch2], axis=-1, name='concat-03')

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
        branch1 = AveragePooling2D(pool_size = (3, 3), strides = (1, 1), padding = 'same', name = base_name + 'ia_branch_1_1')(X_input)
        branch1 = self._conv2d_bn(branch1, filters = 96, kernel_size = (1, 1), strides = (1, 1), padding = 'same', activation='relu', name = base_name + 'ia_branch_1_2')
        
        # Branch 2
        branch2 = self._conv2d_bn(X_input, filters = 96, kernel_size = (1, 1), strides = (1, 1), padding = 'same', activation='relu', name = base_name + 'ia_branch_2_1')
        
        # Branch 3
        branch3 = self._conv2d_bn(X_input, filters = 64, kernel_size = (1, 1), strides = (1, 1), padding = 'same', activation='relu', name = base_name + 'ia_branch_3_1')
        branch3 = self._conv2d_bn(branch3, filters = 96, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation='relu', name = base_name + 'ia_branch_3_2')

        # Branch 4
        branch4 = self._conv2d_bn(X_input, filters = 64, kernel_size = (1, 1), strides = (1, 1), padding = 'same', activation='relu', name = base_name + 'ia_branch_4_1')
        branch4 = self._conv2d_bn(branch4, filters = 96, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation='relu', name = base_name + 'ia_branch_4_2')
        branch4 = self._conv2d_bn(branch4, filters = 96, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation='relu', name = base_name + 'ia_branch_4_3')

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
        branch1 = AveragePooling2D(pool_size = (3, 3), strides = (1, 1), 
                            padding = 'same', name = base_name + 'ib_branch_1_1')(X_input)
        branch1 = self._conv2d_bn(branch1, filters = 128, kernel_size = (1, 1), 
                            strides = (1, 1), padding = 'same', activation='relu', 
                            name = base_name + 'ib_branch_1_2')
        
        # Branch 2
        branch2 = self._conv2d_bn(X_input, filters = 384, kernel_size = (1, 1), 
                            strides = (1, 1), padding = 'same', activation='relu', 
                            name = base_name + 'ib_branch_2_1')
        
        # Branch 3
        branch3 = self._conv2d_bn(X_input, filters = 192, kernel_size = (1, 1), 
                            strides = (1, 1), padding = 'same', activation='relu', 
                            name = base_name + 'ib_branch_3_1')
        branch3 = self._conv2d_bn(branch3, filters = 224, kernel_size = (1, 7), 
                            strides = (1, 1), padding = 'same', activation='relu', 
                            name = base_name + 'ib_branch_3_2')
        branch3 = self._conv2d_bn(branch3, filters = 256, kernel_size = (7, 1), 
                            strides = (1, 1), padding = 'same', activation='relu', 
                            name = base_name + 'ib_branch_3_3')

        # Branch 4
        branch4 = self._conv2d_bn(X_input, filters = 192, kernel_size = (1, 1), 
                            strides = (1, 1), padding = 'same', activation='relu', 
                            name = base_name + 'ib_branch_4_1')
        branch4 = self._conv2d_bn(branch4, filters = 192, kernel_size = (1, 7), 
                            strides = (1, 1), padding = 'same', activation='relu', 
                            name = base_name + 'ib_branch_4_2')
        branch4 = self._conv2d_bn(branch4, filters = 224, kernel_size = (7, 1), 
                            strides = (1, 1), padding = 'same', activation='relu', 
                            name = base_name + 'ib_branch_4_3')
        branch4 = self._conv2d_bn(branch4, filters = 224, kernel_size = (1, 7), 
                            strides = (1, 1), padding = 'same', activation='relu', 
                            name = base_name + 'ib_branch_4_4')
        branch4 = self._conv2d_bn(branch4, filters = 256, kernel_size = (7, 1), 
                            strides = (1, 1), padding = 'same', activation='relu', 
                            name = base_name + 'ib_branch_4_5')

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
        branch1 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name=base_name + 'ic_branch_1_1')(X_input)
        branch1 = self._conv2d_bn(branch1, filters=256, kernel_size=(1, 1),strides=(1, 1), padding='same', activation='relu', name=base_name + 'ic_branch_1_2')

        # Branch 2
        branch2 = self._conv2d_bn(X_input, filters = 256, kernel_size = (1, 1), strides = (1, 1), padding = 'same', activation='relu', name = base_name + 'ic_branch_2_1')
        
        # Branch 3
        branch3 = self._conv2d_bn(X_input, filters = 384, kernel_size = (1, 1), strides = (1, 1), padding = 'same', activation='relu', name = base_name + 'ic_branch_3_1')
        branch3_1 = self._conv2d_bn(branch3, filters = 256, kernel_size = (1, 3), strides = (1, 1), padding = 'same', activation='relu', name = base_name + 'ic_branch_3_2')
        branch3_2 = self._conv2d_bn(branch3, filters = 256, kernel_size = (3, 1), strides = (1, 1), padding = 'same', activation='relu', name = base_name + 'ic_branch_3_3')

        # Branch 4
        branch4 = self._conv2d_bn(X_input, filters = 384, kernel_size = (1, 1), strides = (1, 1), padding = 'same', activation='relu', name = base_name + 'ic_branch_4_1')
        branch4 = self._conv2d_bn(branch4, filters = 448, kernel_size = (1, 3), strides = (1, 1), padding = 'same', activation='relu', name = base_name + 'ic_branch_4_2')
        branch4 = self._conv2d_bn(branch4, filters = 512, kernel_size = (3, 1), strides = (1, 1), padding = 'same', activation='relu', name = base_name + 'ic_branch_4_3')
        branch4_1 = self._conv2d_bn(branch4, filters = 256, kernel_size = (3, 1), strides = (1, 1), padding = 'same', activation='relu', name = base_name + 'ic_branch_4_4')
        branch4_2 = self._conv2d_bn(branch4, filters = 256, kernel_size = (1, 3), strides = (1, 1), padding = 'same', activation='relu', name = base_name + 'ic_branch_4_5')

        # Concatenate branch1, branch2, branch3_1, branch3_2, branch4_1 and branch4_2 along the channel axis
        X = tf.concat(values=[branch1, branch2, branch3_1, branch3_2, branch4_1, branch4_2], axis=3)
        
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
        branch1 = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name= 'ra_branch_1_1')(X_input)

        # Branch 2
        branch2 = self._conv2d_bn(X_input, filters = 384, kernel_size = (3, 3), strides = (2, 2), padding = 'valid', activation='relu', name = 'ra_branch_2_1')
        
        # Branch 3
        branch3 = self._conv2d_bn(X_input, filters = 192, kernel_size = (1, 1), strides = (1, 1), padding = 'same', activation='relu', name = 'ra_branch_3_1')
        branch3 = self._conv2d_bn(branch3, filters = 224, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation='relu', name = 'ra_branch_3_2')
        branch3 = self._conv2d_bn(branch3, filters = 256, kernel_size = (3, 3), strides = (2, 2), padding = 'valid', activation='relu', name = 'ra_branch_3_3')

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
        branch1 = MaxPool2D(pool_size = (3, 3), strides = (2, 2), padding = 'valid', name = 'rb_branch_1_1')(X_input)
        
        # Branch 2
        branch2 = self._conv2d_bn(X_input, filters = 192, kernel_size = (1, 1), strides = (1, 1), padding = 'same', activation='relu', name = 'rb_branch_2_1')
        branch2 = self._conv2d_bn(branch2, filters = 192, kernel_size = (3, 3), strides = (2, 2), padding = 'valid', activation='relu', name = 'rb_branch_2_2')
        
        # Branch 3
        branch3 = self._conv2d_bn(X_input, filters = 256, kernel_size = (1, 1), strides = (1, 1), padding = 'same', activation='relu', name = 'rb_branch_3_1')
        branch3 = self._conv2d_bn(branch3, filters = 256, kernel_size = (1, 7), strides = (1, 1), padding = 'same', activation='relu', name = 'rb_branch_3_2')
        branch3 = self._conv2d_bn(branch3, filters = 320, kernel_size = (7, 1), strides = (1, 1), padding = 'same', activation='relu', name = 'rb_branch_3_3')
        branch3 = self._conv2d_bn(branch3, filters = 320, kernel_size = (3, 3), strides = (2, 2), padding = 'valid', activation='relu', name = 'rb_branch_3_4')

        # Concatenate branch1, branch2 and branch3 along the channel axis
        X = tf.concat(values=[branch1, branch2, branch3], axis=3)
                
        return X


class Unet:
    @staticmethod
    def unet():

        print("hi")


class ResidualAttentionUnet:
    @staticmethod
    def residual_attention_unet():
        print("hello")


class InceptionAttentionUnet:
    @staticmethod
    def inception_attention_unet():
        print("hello")

