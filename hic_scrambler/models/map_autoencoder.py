# Implementing an autoencoder to reconstruct chromosome contact maps from a scrambled version.
# cmdoret, 20210209

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from keras.models import model_from_json
from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Dense,
    Flatten,
    Input,
    MaxPool2D,
    Reshape,
)
import glob
import re
from os.path import join
import numpy as np
from typing import Tuple


def load_data(
    training_path: str = "data/input/training_aa", chunk=0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads training data from several runs, condatenates and normalize it."""
    # Take all sets in training folder

    # Concatenate them into a single array
    x_data = np.load(join(training_path, f"scrambled_chunk_{chunk}.npy"))
    y_data = np.load(join(training_path, f"truth_chunk_{chunk}.npy"))
    if x_data.shape != y_data.shape:
        raise ValueError("Input and output images should have the same shape.")

    # Crop if shape is odd
    if x_data.shape[1] % 2:
        x_data = x_data[:, :-1, :-1]
        y_data = y_data[:, :-1, :-1]
    # Flatten Hi-C images: Divide each image by a diagonal gradient
    # computed across whole dataset
    # x_data = erase_diags(x_data)
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

    def __init__(self, img_size=256):
        super(Unscramble, self).__init__()
        conv_args = {"kernel_size": 3, "activation": "relu", "padding": "same"}
        tconv_args = {
            "kernel_size": 3,
            "strides": 2,
            "activation": "relu",
            "padding": "same",
        }
        self.encoder = tf.keras.Sequential(
            [
                Input(shape=(img_size, img_size, 1)),
                Conv2D(16, **conv_args),
                MaxPool2D((2, 2)),
                Conv2D(8, **conv_args),
                MaxPool2D((2, 2)),
                Conv2D(4, **conv_args),
                MaxPool2D((2, 2)),
                Flatten(),
                Dense(img_size//8 * img_size//8 * 4)
            ]
        )

        self.decoder = tf.keras.Sequential(
            [   Dense(img_size//8 * img_size//8 * 4),
                Reshape((img_size//8, img_size//8, 4)),
                Conv2DTranspose(4, **tconv_args),
                Conv2DTranspose(8, **tconv_args),
                Conv2DTranspose(16, **tconv_args),
                Conv2D(
                    1, kernel_size=1, activation="sigmoid", padding="same",
                ),
            ]
        )

    def call(self, x: np.ndarray) -> np.ndarray:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def load_model(model_dir: str = "data/models/example") -> Model:
    """Loads a trained neural network from a json file"""
    with open(join(model_dir, "model.json"), "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(join(model_dir, "weights.h5"))
    loaded_model.compile(
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        optimizer="adam",
    )
    return loaded_model


def save_model(model: Model, model_dir: str):
    """Saves model configuration and weights to disk."""
    model_json = model.to_json()
    with open(join(model_dir, "model.json"), "w") as json_file:
        json_file.write(model_json)
    model.save_weights(join(model_dir, "weights.h5"))


def train_and_evaluate_model(
    model: Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    model.fit(x_train, y_train, epochs=15)
    validation_results = model.evaluate(x_test, y_test)
    return validation_results


if __name__ == "__main__":
    # Exploration, parameter search, validation etc
    n_folds = 5
    x_data, y_data = load_data()
    x_data = x_data[:, :256, :256, :]
    y_data = y_data[:, :256, :256, :]
    (x_data, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    x_data = x_data.sum(axis=3)[:, :, :, None]
    y_data = x_data + np.random.random((x_data.shape))
    img_size = x_data.shape[1]
    # test_idx = np.random.choice(range(x_data.shape[0]), size=x_data // 5)
    # test_mask, train_mask = np.zeros(x_data.shape[0]), np.ones(x_data.shape[0])
    # test_mask[test_idx] = 1
    # train_mask[test_idx] = 0
    tf.keras.datasets.cifar10.load_data()

    autoencoder = Unscramble(img_size=img_size)
    autoencoder.compile(optimizer="sgd", loss=tf.losses.MeanSquaredError())
    print(
        f'{"-"*10}\nTraining model on {x_data.shape[0]} images of shape {x_data.shape[1]}x{x_data.shape[2]}'
    )
    autoencoder.fit(
        x_data, y_data, epochs=10, shuffle=True, validation_split=0.2
    )
    print(autoencoder.encoder.summary())
    demo_sample = np.random.choice(range(x_data.shape[0]), size=5)
    encoded_imgs = x_data[demo_sample]
    truth_imgs = y_data[demo_sample]
    decoded_imgs = autoencoder.call(x_data[demo_sample]).numpy()

    # Visualize inputs and outputs
    fig, ax = plt.subplots(5, 3, sharex=True, sharey=True)
    for i in range(ax.shape[0]):
        ax[i, 0].imshow(np.log1p(encoded_imgs[i, :, :]))
        ax[i, 1].imshow(np.log1p(decoded_imgs[i, :, :]))
        ax[i, 2].imshow(np.log1p(truth_imgs[i, :, :]))
    plt.suptitle("Random examples of input and decoded samples")
    ax[0, 0].set_title("Input")
    ax[0, 1].set_title("Decoded")
    ax[0, 2].set_title("Truth")
    plt.show()
