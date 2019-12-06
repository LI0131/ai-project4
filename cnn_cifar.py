import os
import keras
import logging
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

logging.basicConfig(level=logging.INFO, filename='out.log')


BATCH_SIZE = os.environ.get('BATCH_SIZE', 32)
NUM_CLASSES = os.environ.get('NUM_CLASSES', 10)
EPOCHS = os.environ.get('EPOCHS', 50)


def run():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                    input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

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
