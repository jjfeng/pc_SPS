import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence

import matplotlib.pyplot as plt

class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        assert latent_dim in [64,128]
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
          layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
          layers.MaxPooling2D((2, 2), padding='same'),
          layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
          layers.MaxPooling2D((2, 2), padding='same'),
          layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
          layers.MaxPooling2D((2, 2), padding='same'),
          layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
          layers.MaxPooling2D((2, 2), padding='same'),
        ])
        self.decoder = tf.keras.Sequential([
          layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
          layers.UpSampling2D((2, 2)),
          layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
          layers.UpSampling2D((2, 2)),
          layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
          layers.UpSampling2D((2, 2)),
          layers.Conv2D(16, (3, 3), activation='relu'),
          layers.UpSampling2D((2, 2)),
          layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def get_autoencoded_imgs(autoencoder, x_test):
    x_test = tf.convert_to_tensor(x_test)
    encoded_imgs = autoencoder.encoder(x_test).eval()
    encoded_imgs = encoded_imgs.reshape((encoded_imgs.shape[0], -1))
    return encoded_imgs

def fit_autoencoder(x_train, x_test, latent_dim: int =32, epochs: int = 10, batch_size: int = 128):
    print(x_train.shape)
    autoencoder = Autoencoder(latent_dim)
    autoencoder.compile(optimizer='adam', loss="binary_crossentropy")

    # Start the autoencoder training off with one epoch of ordinary training
    autoencoder.fit(x_train, x_train,
                    epochs=1,
                    shuffle=True,
                    validation_data=(x_test, x_test))

    # Begin data augmentation
    n_train = len(x_train)
    datagen = ImageDataGenerator(
        rotation_range=360,
        horizontal_flip=True)
    for e in range(epochs):
        print('Epoch', e)
        batches = 0
        for x_batch in datagen.flow(x_train, shuffle=True, batch_size=batch_size):
            batches += 1
            is_batch_final = batches >= n_train / batch_size
            autoencoder.fit(x_batch, x_batch, verbose=1 if batches % 10 == 0 else 0)
            if is_batch_final:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break

        # Plot images for checking
        print("SAVING IMAGES epoch", e)
        decoded_imgs = autoencoder.predict(x_test)

        n = 10
        plt.figure(figsize=(20, 4))
        for i in range(1, n + 1):
            # Display original
            ax = plt.subplot(2, n, i)
            plt.imshow(x_test[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display reconstruction
            ax = plt.subplot(2, n, i + n)
            plt.imshow(decoded_imgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig("_output/mnist_autoencode.png")

    return autoencoder
