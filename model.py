# Trying a simple keras NN to predict SVs in a Hi-C matrix.
# cmdoret, 20190314
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import glob
import re
import numpy as np


def load_data():
    # Take all sets in training folder
    x_data = np.sort(
        [f for f in glob.glob("data/input/training/*") if re.search(r".*x\.npy", f)]
    )
    y_data = np.sort(
        [f for f in glob.glob("data/input/training/*") if re.search(r".*y\.npy", f)]
    )

    # Concatenate them into a single array
    x_data = map(np.load, x_data)
    y_data = map(np.load, y_data)
    x_data = np.vstack(list(x_data))
    y_data = np.hstack(list(y_data))
    x_data = tf.keras.utils.normalize(x_data, axis=1)
    return x_data, y_data


def create_model():
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


def train_and_evaluate_model(model, x_train, y_train, x_test, y_test):
    history = model.fit(x_train, y_train, epochs=5)
    return history


if __name__ == "__main__":
    n_folds = 5
    data, labels = load_data()
    skf = StratifiedKFold(labels, n_splits=n_folds, shuffle=True)

    for i, (train, test) in enumerate(skf):
        print("Running Fold", i + 1, "/", n_folds)
        model = None  # Clearing the NN.
        model = create_model()
        train_and_evaluate_model(
            model, data[train], labels[train], data[test], labels[test]
        )


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

model.evaluate(x_test, y_test, batch_size=16)
