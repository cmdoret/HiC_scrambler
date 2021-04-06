# Implementing an autoencoder to reconstruct chromosome contact maps from a scrambled version.
# The model implementation is heavily based on face-vae: https://github.com/seasonyc/face_vae
# Note: It doesn't work at all for unscrambling
# cmdoret, 20210209

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, model_from_json
from keras.models import model_from_json
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    Input,
    LeakyReLU,
    Reshape,
    UpSampling2D,
    ZeroPadding2D,
)
from os.path import join
import numpy as np
from typing import Tuple, Iterable, Optional


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
            np.load(join(training_path, f"truth_chunk_{chunk}.npy"))[
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


def block_res(x, n_channels: int, kernel_size: int = 3):
    """Residual block"""
    input_x = x
    x = block_conv(x, n_channels, kernel_size=kernel_size)
    x = Conv2D(n_channels, kernel_size, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Add()([input_x, x])

    return x


def block_conv(
    x, n_channels: int, kernel_size: int = 3, padding: str = "same"
):
    """Convolutional block"""
    x = Conv2D(n_channels, kernel_size, padding=padding, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    return x


def block_downsample(x, n_channels: int, kernel_size: int = 4):
    """Downsampling block"""
    x = ZeroPadding2D()(x)
    x = Conv2D(n_channels, kernel_size, strides=(2, 2), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)

    return x


def block_upsample(x, n_channels: int, kernel_size: int = 3):
    """Upsampling block"""
    x = UpSampling2D()(x)
    x = block_conv(x, n_channels, kernel_size=kernel_size)

    return x


def create_encoder(
    img_size: int,
    latent_dim: int = 1024,
    n_channels: int = 16,
    n_layers: int = 5,
) -> Tuple[Model, Tuple[int, int, int]]:
    """Generate encoder part to compress input images to latent space."""
    encoder_input = Input(shape=(img_size, img_size, 1), name="image")
    x = block_conv(encoder_input, n_channels)
    print("First conv:", K.int_shape(x))
    # Compression by successive strided convolutions
    for _ in range(n_layers):
        n_channels *= 2
        x = block_downsample(x, n_channels)
        print("Downsampled:", K.int_shape(x))

    x = block_res(x, n_channels=n_channels)
    conv_shape = K.int_shape(x)
    x = Flatten()(x)
    output = Dense(latent_dim)(x)
    # The output will be flat (1D), but we also provide the shape
    # of the last convolution output to restore the image dimensions later
    print("Encoded:", K.int_shape(x))
    return Model(encoder_input, output, name="encoder"), conv_shape


def create_decoder(
    conv_shape: Tuple[int, int, int],
    latent_dim: int = 1024,
    n_layers: int = 5,
) -> Model:
    """Generate decoder part to reconstruct an image from the latent space."""
    decoder_input = Input(shape=(latent_dim,), name="latent")
    x = Dense(conv_shape[1] * conv_shape[2] * conv_shape[3])(decoder_input)
    x = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
    print("After reshape:", K.int_shape(x))
    n_channels = conv_shape[3]
    x = block_res(x, n_channels)

    # Decompression by successive convolution and upsampling
    for _ in range(n_layers):
        n_channels //= 2
        x = block_upsample(x, n_channels)
        print("Upsampled:", K.int_shape(x))

    x = Conv2D(1, 4, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    print("Final:", K.int_shape(x))
    return Model(decoder_input, x, name="decoder")


def create_autoencoder(
    img_size: int, latent_dim: int, n_layers: int = 5, n_channels: int = 16
):
    encoder, conv_shape = create_encoder(
        img_size, latent_dim, n_layers=n_layers, n_channels=n_channels
    )
    decoder = create_decoder(conv_shape, latent_dim, n_layers=n_layers)
    inp = Input(shape=(img_size, img_size, 1), name="image")
    lat = encoder(inp)
    out = decoder(lat)
    model = Model(inp, out, name="autoencoder")
    return model


def train(
    x_train: np.ndarray,
    x_val: np.ndarray,
    n_layers: int,
    n_channels: int,
    trained_model: Optional[str] = None,
    latent_dim: float = 1024,
    learning_rate: float = 0.0005,
    epochs: int = 100,
):
    """
    Train the autoencoder on input data and return the trained model. A
    pre-trained model can be given to adjust weights, or the model can be
    trained from scratch
    """
    lr_decay_ratio = 0.86
    img_size = x_train.shape[1]

    ae = create_autoencoder(
        img_size, latent_dim, n_channels=n_channels, n_layers=n_layers
    )
    if trained_model is not None:
        trained = load_model(trained_model)
        ae.set_weights(trained.get_weights())
    opt = optimizers.Adam(lr=learning_rate, epsilon=1e-08)

    def ae_loss(x, t_decoded):
        """Total loss for the plain AE"""
        return K.mean(reconstruction_loss(x, t_decoded))

    def reconstruction_loss(x, t_decoded):
        """Reconstruction loss for the plain VAE"""
        return K.sum(
            K.binary_crossentropy(
                K.batch_flatten(x), K.batch_flatten(t_decoded)
            ),
            axis=-1,
        )

    def schedule(epoch, lr):
        if epoch > 0:
            lr *= lr_decay_ratio
        return lr

    lr_scheduler = LearningRateScheduler(schedule, verbose=1)
    ae.compile(optimizer=opt, loss=tf.losses.MeanSquaredError())
    ae.fit(
        x_train,
        x_val,
        epochs=epochs,
        validation_split=0.2,
        verbose=1,
        callbacks=[lr_scheduler],
    )

    return ae


def load_model(path):
    model = model_from_json(open(f"{path}.json", "r").read())
    model.load_weights(f"{path}.h5")
    return model


def save_model(model, path):
    json_string = model.to_json()
    file = open(f"{path}.json", "w")
    file.write(json_string)
    file.close()
    model.save_weights(f"{path}.h5")


def decode_images(x_input, ae):
    """Decode a stack of images"""
    encoded = ae.get_layer("encoder").predict(x_input)
    decoded = ae.get_layer("decoder").predict(encoded)
    return decoded


if __name__ == "__main__":
    # Exploration, parameter search, validation etc
    CROP = 32
    x_data, y_data = load_data(chunks=[0, 1, 2], crop_to=CROP)
    x_data, y_data = np.log1p(x_data), np.log1p(y_data)
    scale_mean, scale_std = x_data.mean(), x_data.std()
    x_data = (x_data - scale_mean) / scale_std
    y_data = (y_data - scale_mean) / scale_std

    x_test, y_test = load_data(chunks=[3], crop_to=CROP)

    # DEBUG
    # (y_data, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    # y_data = y_data.mean(axis=3)[:, :, :, None] / 255
    # x_data = y_data + np.random.random(y_data.shape) / 3

    print(
        f'{"-"*10}\nTraining model on {x_data.shape[0]} images of shape {x_data.shape[1]}x{x_data.shape[2]}'
    )
    autoencoder = train(
        x_data, y_data, latent_dim=1024, n_layers=3, n_channels=16, epochs=25
    )
    print(autoencoder.summary())
    save_model(autoencoder, "data/tmp/weights/map_autoencoder")

    # Prediction on test dataset
    demo_sample = np.random.choice(range(x_test.shape[0]), size=5)
    encoded_imgs = x_test[demo_sample]
    truth_imgs = y_test[demo_sample]
    decoded_imgs = decode_images(x_test[demo_sample], autoencoder)

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

