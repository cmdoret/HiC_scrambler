# Trying a simple keras NN to predict SVs in a Hi-C matrix.
# cmdoret, 20190314
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Initializes a sequential model (i.e. linear stack of layers)
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())  # Flattens input matrix

# NN layer that takes an array of 128 values as input into a 1D array
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
history = model.fit(x_train, y_train, epochs=5)

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
