import os
import keras
import logging
import argparse
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.datasets import cifar10
from keras.losses import binary_crossentropy
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Reshape, UpSampling2D, Input, Lambda

logging.basicConfig(level=logging.INFO, filename='info.log')


BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 32))
NUM_CLASSES = int(os.environ.get('NUM_CLASSES', 10))
EPOCHS = int(os.environ.get('EPOCHS', 50))


def vae(x_train_orig, y_train_orig, x_test, y_test):

    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    image_size = x_train_orig.shape[1]
    original_dim = image_size * image_size
    x_train = np.reshape(x_train_orig, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    inputs = Input(shape=(original_dim, ), name='encoder_input')
    x = Dense(512, activation='relu')(inputs)
    z_mean = Dense(2, name='z_mean')(x)
    z_log_var = Dense(2, name='z_log_var')(x)

    z = Lambda(sampling, output_shape=(2,), name='z')([z_mean, z_log_var])

    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()

    latent_inputs = Input(shape=(2,), name='z_sampling')
    x = Dense(512, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)

    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')

    reconstruction_loss = binary_crossentropy(inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    vae.compile(optimizer='adam')

    vae.fit(
        x_train,
        epochs=100,
        batch_size=128,
        shuffle=True,
        validation_data=(x_test, None),
    )

    predictions = vae.predict(x_train)
    pred = []

    for i in range(len(predictions)):
        if i % 3 == 0:
            r = predictions[i-3].reshape((32, 32, 1))
            g = predictions[i-2].reshape((32, 32, 1))
            b = predictions[i-1].reshape((32, 32, 1))
            pred.append(np.concatenate([r,g,b], axis=2))

    x_exp = []
    y_exp = []
    for i in range(len(pred)):
        x_exp.append(pred[i])
        y_exp.append(y_train_orig[i])

    return np.concatenate([x_train_orig, x_exp], axis=0), np.concatenate([y_train_orig, y_exp], axis=0)


def run():
    parser = argparse.ArgumentParser()
    help_ = "Use VAE to increase the training set size (default=50,000)"
    parser.add_argument("--vae", help=help_, action='store_true')

    args = parser.parse_args()

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    if args.vae:
        # use a variational autoencoder to increase training set size
        x_train, y_train = vae(x_train, y_train, x_test, y_test)

    # convert the image pixel values to a range between 0 and NUM_CLASSES for categorical classification
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES) 
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    # increase the number of filters over time to deal with increasingly complex features
    model = Sequential()
    # same padding pads the input so the output dimensions will be identical
    # using a window size of 3 x 3 with 32 filters
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))

    logging.info(model.summary())

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        shuffle=True
    )

    scores = model.evaluate(x_test, y_test)
    logging.info(f'Test loss: {scores[0]}')
    logging.info(f'Test accuracy: {scores[1]}')


if __name__ == '__main__':
    logging.info('Starting CNN for Cifar10 dataset...')
    run()
