from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.models import Model
import hparams


class AutoEncoder:
    """def __init__(self, img_shape):
    self.img_shape = img_shape"""

    @staticmethod
    def auto_encoder():
        """
        Arguments:
        img_shape_x -- size of the input layer
        code_size -- the size of the hidden representation of the input (code)

        Returns:
        encoder -- keras model for the encoder network
        decoder -- keras model for the decoder network
        """

        # encoder

        inputs = Input(shape=(hparams.IMG_SIZE, hparams.IMG_SIZE, 3))

        y = Conv2D(32, (3, 3), activation="elu", padding="same")(inputs)
        y = MaxPool2D((2, 2), padding="same")(y)

        y = Conv2D(64, (3, 3), activation="elu", padding="same")(y)
        y = MaxPool2D((2, 2), padding="same")(y)

        y = Conv2D(128, (3, 3), activation="elu", padding="same")(y)
        y = MaxPool2D((2, 2), padding="same")(y)

        y = Conv2D(256, (3, 3), activation="elu", padding="same")(y)
        y = MaxPool2D((2, 2), padding="same")(y)

        y = Flatten()(y)

        y = Dense(10, activation="elu")(y)
        out = Dense(1, activation="linear")(y)

        model = Model(inputs=inputs, outputs=out)

        return model
