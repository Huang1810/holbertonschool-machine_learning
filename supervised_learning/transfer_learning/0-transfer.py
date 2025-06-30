#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras as K
import numpy as np


def preprocess_data(X, Y):
    """
    Pre-processes the CIFAR-10 data for the model.

    Args:
        X: numpy.ndarray of shape (m, 32, 32, 3) containing CIFAR-10 data
        Y: numpy.ndarray of shape (m,) containing CIFAR-10 labels

    Returns:
        X_p: numpy.ndarray containing preprocessed X
        Y_p: numpy.ndarray containing preprocessed Y
    """
    # Normalize pixel values to [0, 1]
    X_p = X.astype('float32') / 255.0

    # Convert labels to one-hot encoding
    Y_p = K.utils.to_categorical(Y, 10)

    return X_p, Y_p


def create_model(input_shape=(224, 224, 3), num_classes=10):
    """
    Creates a transfer learning model using ResNet50.

    Args:
        input_shape: tuple, input image shape
        num_classes: int, number of output classes

    Returns:
        model: compiled Keras model
    """
    # Load ResNet50 with ImageNet weights, excluding top layers
    base_model = K.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze all layers except the last 4
    for layer in base_model.layers[:-4]:
        layer.trainable = False

    # Create new model
    inputs = K.Input(shape=(32, 32, 3))
    # Upscale images to 224x224 for ResNet50
    x = K.layers.Lambda(lambda x: tf.image.resize(x, [224, 224]))(inputs)
    x = base_model(x)

    # Add custom layers
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dense(256, activation='relu')(x)
    x = K.layers.Dropout(0.5)(x)
    outputs = K.layers.Dense(num_classes, activation='softmax')(x)

    model = K.Model(inputs, outputs)

    # Compile model
    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def main():
    # Load and preprocess CIFAR-10 data
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train_p, Y_train_p = preprocess_data(X_train, Y_train)
    X_test_p, Y_test_p = preprocess_data(X_test, Y_test)

    # Create model
    model = create_model()

    # Compute features for frozen layers once
    X_train_features = tf.image.resize(X_train_p, [224, 224])
    X_test_features = tf.image.resize(X_test_p, [224, 224])

    # Train model
    model.fit(
        X_train_features,
        Y_train_p,
        batch_size=128,
        epochs=20,
        validation_data=(X_test_features, Y_test_p),
        callbacks=[
            K.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            K.callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=3
            )
        ]
    )

    # Save compiled model
    model.save('cifar10.h5')


if __name__ == '__main__':
    main()
