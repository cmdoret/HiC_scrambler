# Implementing an autoencoder to reconstruct chromosome contact maps from a scrambled version.
# cmdoret, 20210209

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Model
from keras.models import model_from_json
from keras.utils.vis_utils import plot_model
import glob
import re
from os.path import join
import numpy as np


def load_data(training_path="data/input/training"):
    """Loads training data from several runs, condatenates and normalize it."""
    # Take all sets in training folder
    x_data = np.sort(
        [
            f
            for f in glob.glob(training_path + "/*")
            if re.search(r".*x\.npy", f)
        ]
    )
    y_data = np.sort(
        [
            f
            for f in glob.glob(training_path + "/*")
            if re.search(r".*y\.npy", f)
        ]
    )

    # Concatenate them into a single array
    x_data = map(np.load, x_data)
    y_data = map(np.load, y_data)
    x_data = np.vstack(list(x_data))
    y_data = np.hstack(list(y_data))
    x_data = tf.keras.utils.normalize(x_data, axis=1)
    # Flatten Hi-C images: Divide each image by a diagonal gradient
    # computed across whole dataset
    x_data = erase_diags(x_data)
    # X must be 4D (for convolutional layers) and made up of floats
    x_data = x_data[:, :, :, None].astype(float)
    return x_data, y_data


def erase_diags(imgs):
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

    def __init__(self):
        super(Unscramble, self).__init__()
        self.encoder = tf.keras.Sequential(
            [
                tf.layers.Input(shape=(28, 28, 1)),
                tf.layers.Conv2D(
                    16, (3, 3), activation="relu", padding="same", strides=2
                ),
                tf.layers.Conv2D(
                    8, (3, 3), activation="relu", padding="same", strides=2
                ),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.layers.Conv2DTranspose(
                    8,
                    kernel_size=3,
                    strides=2,
                    activation="relu",
                    padding="same",
                ),
                tf.layers.Conv2DTranspose(
                    16,
                    kernel_size=3,
                    strides=2,
                    activation="relu",
                    padding="same",
                ),
                tf.layers.Conv2D(
                    1, kernel_size=(3, 3), activation="sigmoid", padding="same"
                ),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def load_model(model_dir="data/models/example"):
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


def save_model(model, model_dir):
    """Saves model configuration and weights to disk."""
    model_json = model.to_json()
    with open(join(model_dir, "model.json"), "w") as json_file:
        json_file.write(model_json)
    model.save_weights(join(model_dir, "weights.h5"))


def train_and_evaluate_model(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train, epochs=15)
    validation_results = model.evaluate(x_test, y_test)
    return validation_results


if __name__ == "__main__":
    # Exploration, parameter search, validation etc
    n_folds = 5
    x_data, y_data = load_data()
    img_size = x_data.shape[1]
    n_labels = len(np.unique(y_data))
    test_idx = np.random.choice(range(x_data.shape[0]), size=x_data // 5)
    # test_mask, train_mask = np.zeros(x_data.shape[0]), np.ones(x_data.shape[0])
    # test_mask[test_idx] = 1
    # train_mask[test_idx] = 0

    autoencoder = Unscramble()
    autoencoder.compile(optimizer="adam", loss=tf.losses.MeanSquaredError())
    autoencoder.fit(
        x_data, y_data, epochs=10, shuffle=True, validation_split=0.2
    )
    autoencoder.encoder.summary()
    encoded_imgs = autoencoder.encoder(x_data).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
    print(
        f'{"-"*10}\nTraining model on {x_data.shape[0]} images of shape {x_data.shape[1]}x{x_data.shape[2]}'
    )
    print(autoencoder.summary())

    # Visualize inputs
    fig, ax = plt.subplots(5, 5, sharex=True, sharey=True)
    label_mapping = {0: "NORMAL", 1: "INV", 2: "DEL"}
    sel_id = np.random.choice(range(x_data.shape[0]), size=25)
    for i, a in zip(sel_id, ax.flat):
        a.imshow(np.log1p(x_data[i, :, :]))
        a.set_title(label_mapping[y_data[i]])
    plt.suptitle("Random examples of input samples")
    plt.show()
