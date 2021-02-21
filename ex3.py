from keras.utils import to_categorical
from keras.layers.merge import concatenate
from keras.layers import Input, BatchNormalization
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
import numpy as np
from numpy import vstack
from numpy.random import randn, randint
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import LeakyReLU
import tensorflow.keras.layers as layers
from collections import defaultdict
import pandas as pd
import os
import math

latent_dim = 64
noise_sigma = 0.35
train_AE = False
train_gen = False
train_con_gen = False

sml_train_size = 50
noise_dim = 32
num_classes = 10


def set_style():
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.facecolor'] = '00000007'
    plt.rcParams['axes.edgecolor'] = '0000003D'
    plt.rcParams['axes.labelcolor'] = '000000D9'
    plt.rcParams['xtick.color'] = '000000'
    plt.rcParams['ytick.color'] = '000000'
    plt.rcParams['legend.facecolor'] = 'FFFFFFD9'
    plt.rcParams['legend.edgecolor'] = '000000D9'
    plt.rcParams['figure.facecolor'] = 'FFFFFF'
    plt.rcParams['savefig.facecolor'] = 'FFFFFF'
    plt.rcParams['figure.figsize'] = 12, 8


def AE():
    encoder = Sequential()
    encoder.add(layers.Conv2D(16, (4, 4), strides=(2, 2), activation='relu', padding='same', input_shape=(32, 32, 1)))
    encoder.add(layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    encoder.add(layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    encoder.add(layers.Conv2D(96, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    encoder.add(layers.Reshape((2 * 2 * 96,)))
    encoder.add(layers.Dense(latent_dim))
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    decoder = Sequential()
    decoder.add(layers.Dense(2 * 2 * 96, activation='relu', input_shape=(latent_dim,)))
    decoder.add(layers.Reshape((2, 2, 96)))
    decoder.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    decoder.add(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    decoder.add(layers.Conv2DTranspose(16, (4, 4), strides=(2, 2), activation='relu', padding='same'))
    decoder.add(layers.Conv2DTranspose(1, (4, 4), strides=(2, 2), activation='sigmoid', padding='same'))
    autoencoder = keras.Model(encoder.inputs, decoder(encoder.outputs))
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder, encoder, decoder


def test_ae():
    latent_codes = encoder.predict(x_test)
    decoded_imgs = decoder.predict(latent_codes)
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(1, n + 1):
        # Display original
        ax = plt.subplot(2, n, i)
        plt.imshow(x_test[i].reshape(32, 32))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + n)
        plt.imshow(decoded_imgs[i].reshape(32, 32))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


# -------------- Classifier --------------

def compare_result_test(models_sets_map, y_test):
    d = defaultdict(list)
    for model, X_test in models_sets_map.values():
        model_results = model.evaluate(X_test, y_test, verbose=0)
        for k, metric in enumerate(model.metrics_names):
            d[metric].append(model_results[k])

    df = pd.DataFrame(d, index=[name for name in models_sets_map.keys()])
    ax = df.plot(kind='bar', width=.7)
    for p in ax.patches:
        ax.annotate(np.round(p.get_height(), decimals=3),
                    (p.get_x() + p.get_width() / 2., p.get_height()), ha='center',
                    va='center', xytext=(0, 10), textcoords='offset points')
    plt.title('Test Results')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.show()


def fit_and_compare_classifiers():
    classifier = Sequential()
    classifier.add(layers.Dense(64, activation='elu', input_shape=(latent_dim,)))
    classifier.add(layers.Dense(32, activation='elu'))
    classifier.add(layers.Dense(10, activation='softmax'))

    train_codes = encoder.predict(x_train[:sml_train_size])
    test_codes = encoder.predict(x_test)
    classifier.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    classifier.fit(train_codes, y_train[:sml_train_size],
                   epochs=120,
                   batch_size=16,
                   shuffle=True,
                   validation_data=(test_codes, y_test))

    full_cls_enc = keras.models.clone_model(encoder)
    full_cls_cls = keras.models.clone_model(classifier)
    full_cls = keras.Model(full_cls_enc.inputs, full_cls_cls(full_cls_enc.outputs))

    full_cls.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    full_cls.fit(x_train[:sml_train_size], y_train[:sml_train_size],
                 epochs=120,
                 batch_size=16,
                 shuffle=True,
                 validation_data=(x_test, y_test))
    compare_result_test({'No Encoder Classifier': (full_cls, x_test),
                         'Encoder Classifier': (classifier, test_codes)}, y_test)

    return classifier, full_cls


# -------------- GAN --------------

def discriminator():
    model = Sequential()
    model.add(layers.Dense(128, input_shape=(latent_dim,)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(layers.Dense(64))
    model.add(LeakyReLU(alpha=0.2))
    model.add(layers.Dense(32))
    model.add(LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def generator():
    model = Sequential()

    model.add(layers.Dense(64, input_shape=(noise_dim,)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(layers.Dense(128))
    model.add(LeakyReLU(alpha=0.2))
    model.add(layers.Dense(128))
    model.add(LeakyReLU(alpha=0.2))
    model.add(layers.Dense(latent_dim))

    return model


def gan(g_model, d_model):
    d_model.trainable = False
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def sample_images(g_model, n_samples, noise_dim=32, uniform=False):
    noise = sample_noise(n_samples, noise_dim, uniform)
    gen_imgs = g_model.predict(noise)
    imgs = decoder.predict(gen_imgs)
    n = int(math.sqrt(n_samples))

    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(imgs[i, :, :, 0], cmap='gray')
    plt.show()


def train(g_model, d_model, gan_model, noise_dim=32, n_epochs=100, n_batch=256):
    epoch_n_iter = x_train.shape[0] // n_batch
    batch_size = n_batch // 2
    fake_class = np.zeros(shape=(batch_size, 1))
    real_class = np.ones(shape=(batch_size, 1))

    for i in range(n_epochs):

        for j in range(epoch_n_iter):
            idx = randint(0, x_train.shape[0], batch_size)
            noise = sample_noise(batch_size, noise_dim)

            X_real = encoder.predict(x_train[idx])
            X_fake = g_model.predict(noise)

            # create train set for the discriminator and train it
            X, y = vstack((X_real, X_fake)), vstack((real_class, fake_class))
            d_loss, d_acc = d_model.train_on_batch(X, y)

            # train GAN model
            g_loss, _ = gan_model.train_on_batch(noise, real_class)
            print(' {:d}  {:d}/{:d} discriminator loss: {:.2f} generator loss: {:.2f} acc {:.2f}'.format(i + 1, j + 1,
                                                                                                         epoch_n_iter,
                                                                                                         d_loss, g_loss,
                                                                                                         d_acc))
        sample_images(g_model, batch_size)


def plot_interpolation(generator, encoder, decoder):
    n = 10
    plt.figure(figsize=(20, 4))

    noise = tf.random.normal([2, noise_dim])
    L1_g, L2_g = generator.predict(noise)
    L1_e, L2_e = encoder(x_test[:2])
    weights = np.linspace(0, 1, 10)

    inter_gan = decoder.predict(np.array([a * L1_g + (1 - a) * L2_g for a in weights]))
    encoder_inter = decoder.predict(np.array([a * L1_e + (1 - a) * L2_e for a in weights]))

    plot_from_two_models(encoder_inter, inter_gan, n)


def plot_from_two_models(imgs_1, imgs_2, n):
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(imgs_1[i].reshape(32, 32))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + n + 1)
        plt.imshow(imgs_2[i].reshape(32, 32))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


# -------------- Conditional GAN --------------

def conditional_discriminator():
    inputs = layers.Input(shape=(latent_dim,))
    labels = layers.Input(shape=(10,))
    x = concatenate([inputs, labels], axis=1)
    x = layers.Dense(128)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(64)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(32)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model([inputs, labels], x, name='discriminator')
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def conditional_gen():
    inputs = Input(shape=(noise_dim,), name='z')
    labels = Input(shape=(10,), name='labels')

    x = concatenate([inputs, labels], axis=1)
    x = layers.Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)

    x = layers.Dense(257, activation='relu')(x)
    x = BatchNormalization()(x)

    x = layers.Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)

    x = layers.Dense(latent_dim)(x)
    model = keras.Model([inputs, labels], x, name='generator')

    return model


def conditional_gan(g_model, d_model):
    d_model.trainable = False
    inputs = layers.Input(shape=(32,), )
    labels = layers.Input(shape=(10,), )

    x1 = g_model([inputs, labels])
    x2 = d_model([x1, labels])
    model = keras.Model([inputs, labels], x2)

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def sample_classes_images(g_model, noise_dim=32, num_classes=10, digit=None, uniform=False):
    noise = sample_noise(num_classes, noise_dim, uniform)

    class_num = np.arange(0, num_classes).reshape(-1, 1)
    gen_imgs = g_model.predict([noise, to_categorical(class_num)])
    imgs = decoder.predict(gen_imgs)
    n_rows, n_columns = 2, num_classes // 2
    fig, axs = plt.subplots(n_rows, n_columns)
    cur_class = 0
    for i in range(n_rows):
        for j in range(n_columns):
            axs[i, j].imshow(imgs[cur_class, :, :, 0], cmap='gray')
            axs[i, j].set_title(class_num[cur_class])
            axs[i, j].axis('off')
            cur_class += 1
    plt.show()


def sample_specific_class(g_model, N=10, noise_dim=32, num_classes=10, digit=0, uniform=False):
    class_num = np.arange(0, num_classes).reshape(-1, 1)
    fig, axs = plt.subplots(1, N)

    for k in range(N):
        noise = sample_noise(num_classes, noise_dim, uniform)
        gen_imgs = g_model.predict([noise, to_categorical(class_num)])
        imgs = decoder.predict(gen_imgs)
        axs[k].imshow(imgs[digit, :, :, 0], cmap='gray')
        axs[k].set_title(class_num[digit])
        axs[k].axis('off')
    plt.show()


def sample_noise(batch_size, noise_dim, uniform=False):
    if uniform:
        return np.random.uniform(-1.0, 1.0, size=[batch_size, noise_dim])
    return tf.random.normal([batch_size, noise_dim])


def train_conditional(g_model, d_model, gan_model, noise_dim=32, n_epochs=100, n_batch=256):
    epoch_n_iter = x_train.shape[0] // n_batch
    batch_size = n_batch // 2
    fake_class = np.zeros(shape=(batch_size, 1))
    real_class = np.ones(shape=(batch_size, 1))

    for i in range(n_epochs):

        for j in range(epoch_n_iter):
            idx = randint(0, x_train.shape[0], batch_size)
            noise = sample_noise(batch_size, noise_dim)

            X_real = encoder.predict(x_train[idx])
            real_labels = y_train[idx]
            X_fake = g_model.predict([noise, real_labels])
            fake_labels = np.eye(num_classes)[np.random.choice(num_classes, batch_size)]

            # create train set for the discriminator and train it
            labels = vstack((real_labels, real_labels))
            X, y = vstack((X_real, X_fake)), vstack((real_class, fake_class))
            d_loss, d_acc = d_model.train_on_batch([X, labels], y)

            # train GAN model
            g_loss, _ = gan_model.train_on_batch([noise, fake_labels], real_class)
            print(' {:d}  {:d}/{:d} discriminator loss: {:.2f} generator loss: {:.2f} acc {:.2f}'.format(i + 1, j + 1,
                                                                                                         epoch_n_iter,
                                                                                                         d_loss, g_loss,
                                                                                                         d_acc))

        sample_classes_images(g_model)


def train_ae():
    os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True)
    autoencoder.fit(x_train + noise_sigma * np.random.randn(*x_train.shape), x_train,
                    epochs=15,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test, x_test),
                    callbacks=[cp_callback])


if __name__ == '__main__':
    set_style()
    # load train and test images, and pad & reshape them to (-1,32,32,1)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1)).astype('float32') / 255.0
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1)).astype('float32') / 255.0
    x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)))
    x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)))

    y_train = keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')
    y_test = keras.utils.to_categorical(y_test, num_classes=10, dtype='float32')

    autoencoder, encoder, decoder = AE()
    checkpoint_path = "model_save/cp.ckpt"

    if train_AE:
        train_ae()
    else:
        autoencoder.load_weights(checkpoint_path)
    test_ae()
    classifier,classifier_copy = fit_and_compare_classifiers()

    gen_path = "model_save/gan.h5"
    d_model = discriminator()
    g_model = generator()
    gan_model = gan(g_model, d_model)

    if train_gen:
        train(g_model, d_model, gan_model)
        gan_model.save(gen_path)

    else:
        gan_model.load_weights(gen_path)
        sample_images(g_model, 25)

    plot_interpolation(g_model, encoder, decoder)

    cgen_path = "model_save/con_gan.h5"
    cd_model = conditional_discriminator()
    cg_model = conditional_gen()
    cgan_model = conditional_gan(cg_model, cd_model)

    if train_con_gen:
        train_conditional(cg_model, cd_model, cgan_model, n_epochs=35)
        cgan_model.save(cgen_path)
    else:
        cgan_model.load_weights(cgen_path)
        for i in range(10):
            sample_specific_class(cg_model, num_classes=10, digit=i)
        sample_classes_images(cg_model, num_classes=10)
