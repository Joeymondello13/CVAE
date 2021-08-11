import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from dataset import train_valid_split, ContactMapDataset
import h5py
import csv
from sklearn.manifold import TSNE
from typing import TypeVar, Type, Union, Optional, Dict, Any
from colordict import *

PathLike = Union[str, Path]

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def encoder(latent_dim = 3, verbose = True):
    encoder_inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(64, 3, activation="relu", strides=1, padding="same")(encoder_inputs)
    x = layers.Conv2D(64, 3, activation="relu", strides=1, padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(64, 3, activation="sigmoid", strides=1, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.0)(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    if verbose:encoder.summary()
    return encoder

def decoder(latent_dim = 3, verbose = True):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()
    return decoder

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def build_model():
    encoder_model = encoder(verbose=False)
    decoder_model = decoder(verbose=False)
    vae = VAE(encoder_model, decoder_model)
    vae.compile(optimizer=keras.optimizers.Adagrad())
    return vae

def make_dataset(input_path):
    dataset = ContactMapDataset(
        path=input_path,
        shape=(1, 612, 612),
        dataset_name='contact_map',
        scalar_dset_names=[],
        values_dset_name=None,
        scalar_requires_grad=False,
        in_memory=True,
    )

    split_pct = 0.8
    train_loader, valid_loader = train_valid_split(
        dataset,
        split_pct,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        pin_memory=True,
    )
    ignore_gpu = False
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and not ignore_gpu else "cpu"
    )

    data = None
    for batch in train_loader:
        x = batch["X"].to(device).flatten()[:-16]
        y = x.reshape(30575, 28, 28).cpu()
        print(y.shape)
        data = np.vstack((data, y)) if not data is None else y

    N = len(data)
    data = data.reshape((N, 28, 28, 1))
    train_idx = int(N * 0.8)
    train_data, test_data = data[:train_idx], data[train_idx:]
    return train_data, test_data

def plot_label_clusters(vae,data,label):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = vae.encoder.predict(data)
    plt.figure()
    plt.scatter(z_mean[:, 0], z_mean[:, 1],c=label)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()
    #fig = plt.figure(figsize=(4, 4))

    #ax = fig.add_subplot(111, projection='3d')
    #plt.show()
def plot_loss(csv):
    big = [0, 30, 50, 70, 90, 110, 130, 150]
    data = pd.read_csv(csv, delimiter=',')
    loss = data['loss']
    plt.figure()
    plt.xlabel('Epochs',fontsize='medium')
    plt.xticks(big)
    plt.ylabel('Loss',fontsize='medium')
    plt.plot(loss,color='red')
    plt.savefig('loss_plot')

def log_latent_visualization(
    data: np.ndarray,
    colors: Dict[str, np.ndarray],
    output_path: PathLike,
    epoch: int = 0,
    n_samples: Optional[int] = None,
    method: str = "TSNE",
) -> Dict[str, str]:
    from plotly.io import to_html

    # Make temp variables to not mutate input data
    if n_samples is not None:
        inds = np.random.choice(len(data), n_samples)
        _data = data[inds]
        _colors = {name: color[inds] for name, color in colors.items()}
    else:
        _data = data
        _colors = colors

    if method == "PCA":
        from sklearn.decomposition import PCA

        model = PCA(n_components=3)
        data_proj = model.fit_transform(_data)

    elif method == "TSNE":
        try:
            # Attempt to use rapidsai
            from cuml.manifold import TSNE

            # rapidsai only supports 2 dimensions
            model = TSNE(n_components=2, method="barnes_hut")
        except ImportError:
            from sklearn.manifold import TSNE

            model = TSNE(n_components=3, n_jobs=1)

        data_proj = model.fit_transform(_data)

    elif method == "LLE":
        from sklearn import manifold

        data_proj, _ = manifold.locally_linear_embedding(
            _data, n_neighbors=12, n_components=3
        )
    else:
        raise ValueError(f"Invalid dimensionality reduction method {method}")

    html_strings = {}
    for color in _colors:
        fig = plot_scatter(data_proj, _colors, color)
        html_string = to_html(fig)
        html_strings[color] = html_string

        fname = Path(output_path) / f"latent_space-{method}-{color}-epoch-{epoch}.html"
        with open(fname, "w") as f:
            f.write(html_string)

    return html_strings

def plot_scatter(
    data: np.ndarray,
    color_dict: Dict[str, np.ndarray] = {},
    color: Optional[str] = None,
):

    import pandas as pd
    import plotly.express as px

    df_dict = color_dict.copy()

    dim = data.shape[1]
    assert dim in [2, 3]
    for i, name in zip(range(dim), ["x", "y", "z"]):
        df_dict[name] = data[:, i]

    df = pd.DataFrame(df_dict)
    scatter_kwargs = dict(
        x="x",
        y="y",
        color=color,
        width=1000,
        height=1000,
        size_max=7,
        hover_data=list(df_dict.keys()),
    )
    if dim == 2:
        fig = px.scatter(df, **scatter_kwargs)
    else:  # dim == 3
        fig = px.scatter_3d(df, z="z", **scatter_kwargs)
    return fig

if __name__ == '__main__':

    # Load training and validation data
    input_path = 'dataset/contact_maps.h5'

    train_data, test_data = make_dataset(input_path)

    np.save('dataset/train_data.npy', train_data)
    np.save('dataset/test_data.npy', test_data)

    train_data = np.load('dataset/train_data.npy')
    test_data = np.load('dataset/test_data.npy')
    data = train_data.reshape(293520,784)
    test = data[:500]
    colors = {'240': 'red'}
    log_latent_visualization(data=test,output_path='dataset',epoch=5,colors=colors)

    # vae = build_model()
    # history = vae.fit(train_data,epochs=1,batch_size=64)
    # dtf = pd.DataFrame(history.history)
    # dtf.to_csv('loss.csv')
    # plot_loss('loss.csv')

    #vae.save('model')




