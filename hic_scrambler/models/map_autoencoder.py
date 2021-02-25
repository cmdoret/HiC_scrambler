# Implementing an autoencoder to reconstruct chromosome contact maps from a scrambled version.
# Note: It doesn't work at all for unscrambling
# cmdoret, 20210209

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from keras.models import model_from_json
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Flatten,
    Input,
    Reshape,
)
import glob
import re
from os.path import join
import numpy as np
from typing import Tuple, Iterable


def load_data(
    training_path: str = "data/input/training_aa",
    chunks: Iterable[int] = [0],
    crop_to: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads training data from several runs, condatenates and normalize it."""
    # Take all sets in training folder

    # Concatenate them into a single array
    x_data = np.vstack(
        [
            np.load(join(training_path, f"scrambled_chunk_{chunk}.npy"))[
                :, :crop_to, :crop_to
            ]
            for chunk in chunks
        ]
    )
    y_data = np.vstack(
        [
            np.load(join(training_path, f"scrambled_chunk_{chunk}.npy"))[
                :, :crop_to, :crop_to
            ]
            for chunk in chunks
        ]
    )
    if x_data.shape != y_data.shape:
        raise ValueError("Input and output images should have the same shape.")

    # Crop if shape is odd
    if x_data.shape[1] % 2:
        x_data = x_data[:, :-1, :-1]
        y_data = y_data[:, :-1, :-1]
    # Flatten Hi-C images: Divide each image by a diagonal gradient
    # computed across whole dataset
    x_data = erase_diags(x_data)
    y_data = erase_diags(y_data)
    # X must be 4D (for convolutional layers) and made up of floats
    x_data = x_data[:, :, :, None].astype(float)
    y_data = y_data[:, :, :, None].astype(float)
    return x_data, y_data


def erase_diags(imgs: np.ndarray) -> np.ndarray:
    """
    Given a stack of N images of shape x,y of shape (N,x,y), divide each
    pixel of each image by the average of its diagonal across the whole stack
    """
    # Mean of each pixel across all imgs
    avg_img = imgs.mean(axis=0)
    # Mean of each diagonal across stack
    avg_dia = [
        np.mean(np.diagonal(avg_img, d)) for d in range(avg_img.shape[0])
    ]
    # Build matrix as a smooth diagonal gradient
    avg_grd = np.zeros(avg_img.shape)
    for k, v in enumerate(avg_dia):
        avg_grd += np.diag([v for i in range(avg_img.shape[0] - k)], k=k)
    # Make gradient symmetric
    avg_grd = avg_grd + np.transpose(avg_grd) - np.diag(np.diag(avg_grd))
    # Divide each image in the stack by gradient
    return imgs / avg_grd


# Inherits from model, so it has the standard tf methods (fit, evaluate, ...)
class Unscramble(Model):
    """Autoencoder to generate correct maps from scrambled maps (i.e. with SV)"""

    def __init__(self, img_size=256, latent_dim=1024, n_layers=5, channels=32):
        super(Unscramble, self).__init__()

        # Build downsampling layers iteratively
        self.encoder = tf.keras.Sequential(
            [Input(shape=(img_size, img_size, 1)),]
        )
        for i in range(n_layers):
            self.encoder.add(
                Conv2D(
                    channels,
                    kernel_size=3,
                    strides=2,
                    activation="relu",
                    padding="same",
                )
            )
            self.encoder.add(BatchNormalization())
            channels *= 2

        # Bottleneck with a couple dense layers
        self.encoder.add(Flatten())
        self.encoder.add(Dense(latent_dim))

        self.decoder = tf.keras.Sequential([Input(shape=(latent_dim,))])

        # Compute dimension of the image after the last downsampling layer
        conv_dim = (
            img_size // (2 ** n_layers),
            img_size // (2 ** n_layers),
            channels,
        )
        self.decoder.add(Dense(conv_dim[0] * conv_dim[1] * conv_dim[2]))
        self.decoder.add(Reshape(conv_dim))

        # Build upsampling layers iteratively
        for i in range(n_layers):
            self.decoder.add(
                Conv2DTranspose(
                    channels,
                    kernel_size=3,
                    strides=2,
                    activation="relu",
                    padding="same",
                )
            )
            channels /= 2

        self.decoder.add(
            Conv2D(1, kernel_size=1, activation="relu", padding="same")
        )
        self.encoder.add(BatchNormalization())

    def call(self, x: np.ndarray) -> np.ndarray:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    # Exploration, parameter search, validation etc
    n_folds = 5
    x_data, y_data = load_data(chunks=[0, 1, 2, 3], crop_to=128)
    x_data, y_data = np.log1p(x_data), np.log1p(y_data)
    # x_data = x_data[:, :32, :32, :]
    # y_data = y_data[:, :32, :32, :]
    scale_mean, scale_std = x_data.mean(), x_data.std()
    x_data = (x_data - scale_mean) / scale_std
    y_data = (y_data - scale_mean) / scale_std
    # y_data = x_data.copy()
    # y_data[:, 16:, :, :] = y_data[:, 16:, ::-1, :]
    # (y_data, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    # y_data = y_data.mean(axis=3)[:, :, :, None] / 255
    # x_data = y_data + np.random.random(y_data.shape) / 3
    img_size = x_data.shape[1]
    autoencoder = Unscramble(
        img_size=img_size, latent_dim=1024, n_layers=3, channels=16
    )
    autoencoder.compile(optimizer="adam", loss=tf.losses.MeanSquaredError())
    print(
        f'{"-"*10}\nTraining model on {x_data.shape[0]} images of shape {x_data.shape[1]}x{x_data.shape[2]}'
    )
    autoencoder.fit(
        x_data, y_data, epochs=100, shuffle=True, validation_split=0.2
    )
    print(autoencoder.encoder.summary())
    demo_sample = np.random.choice(range(x_data.shape[0]), size=5)
    encoded_imgs = x_data[demo_sample]
    truth_imgs = y_data[demo_sample]
    decoded_imgs = autoencoder.call(x_data[demo_sample]).numpy()

    # Visualize inputs and outputs
    fig, ax = plt.subplots(5, 3, sharex=True, sharey=True)
    for i in range(ax.shape[0]):
        ax[i, 0].imshow(encoded_imgs[i, :, :, 0])
        ax[i, 1].imshow(decoded_imgs[i, :, :, 0])
        ax[i, 2].imshow(truth_imgs[i, :, :, 0])
    plt.suptitle("Random examples of input and decoded samples")
    ax[0, 0].set_title("Input")
    ax[0, 1].set_title("Decoded")
    ax[0, 2].set_title("Truth")
    plt.show()
