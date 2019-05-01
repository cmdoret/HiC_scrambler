# Trying a simple keras NN to predict SVs in a Hi-C matrix.
# cmdoret, 20190314
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from keras.models import model_from_json
import glob
import re
from os.path import join
import numpy as np


def load_data(training_path="data/input/training"):
    """Loads training data from several runs, condatenates and normalize it."""
    # Take all sets in training folder
    x_data = np.sort(
        [f for f in glob.glob(training_path + "/*") if re.search(r".*x\.npy", f)]
    )
    y_data = np.sort(
        [f for f in glob.glob(training_path + "/*") if re.search(r".*y\.npy", f)]
    )

    # Concatenate them into a single array
    x_data = map(np.load, x_data)
    y_data = map(np.load, y_data)
    x_data = np.vstack(list(x_data))
    y_data = np.hstack(list(y_data))
    x_data = tf.keras.utils.normalize(x_data, axis=1)
    return x_data, y_data


def create_model():
    """Builds model from scratch for training"""
    # Initializes a sequential model (i.e. linear stack of layers)
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Flatten())  # Flattens input matrix

    # NN layer that takes an array of 128 values as input into a 1D array
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def load_model(model_dir="data/models/example"):
    """Loads a trained neural network from a json file"""
    with open(join(model_dir, "model.json"), "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(join(model_dir, "weights.h5"))
    loaded_model.compile(
        loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer="adam"
    )
    return loaded_model


def save_model(model, model_dir):
    """Saves model configuration and weights to disk."""
    model_json = model.to_json()
    with open(join(model_dir, "model.json"), "w") as json_file:
        json_file.write(model_json)
    model.save_weigths(join(model_dir, "weights.h5"))


def train_and_evaluate_model(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train, epochs=15)
    validation_results = model.evaluate(x_test, y_test)
    return validation_results


if __name__ == "__main__":
    n_folds = 5
    x_data, y_data = load_data()
    # Splits data into kfolds for cross validation
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True)
    kfold_acc, kfold_loss = [None] * n_folds, [None] * n_folds
    for i, (train, test) in enumerate(kfold.split(x_data, y_data)):
        print("Running Fold", i + 1, "/", n_folds)
        model = None  # Clearing the NN.
        model = create_model()
        kfold_loss[i], kfold_acc[i] = train_and_evaluate_model(
            model, x_data[train], y_data[train], x_data[test], y_data[test]
        )
# Loss and accuracy of each fold

plt.bar(range(len(kfold_loss)), height=kfold_loss)
plt.show()
plt.bar(range(len(kfold_acc)), height=kfold_acc)
plt.show()

# On all data at once
history = model.fit(x_data, y_data, epochs=15)
# Plot training & validation accuracy values
plt.plot(history.history["acc"])
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper left")
plt.show()

# Plot training & validation loss values
plt.plot(history.history["loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper left")
plt.show()

