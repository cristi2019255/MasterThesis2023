# Copyright 2023 Cristian Grosu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import numpy as np

from ..AbstractNN import AbstractNN, SEED
from ...Logger import LoggerInterface, LoggerModel

DECODER_NAME = "decoder"
ENCODER_NAME = "encoder"
SSNP_NAME = "ssnp"
CLASSIFIER_NAME = "classifier"

DECODER_LOSS = "mean_squared_error"
CLASSIFIER_LOSS = "sparse_categorical_crossentropy"
DECODER_LOSS_WEIGHT = 1.0
CLASSIFIER_LOSS_WEIGHT = 0.125


class SSNP(AbstractNN):
    def __init__(self,
                 folder_path: str,
                 logger: LoggerInterface | None = None):
        """
            Creates an SSNP model.

            Args:
                folder_path: The path to the folder where the model will be saved/loaded.
                logger: The logger used to log the model's progress.
        """
        super().__init__(folder_path=folder_path, logger=logger, nn_name=SSNP_NAME)

    def __build__(self, input_shape: tuple = (28, 28), num_classes: int = 10, show_summary: bool = False, is_data_normalized: bool = True):
        """
            Assembles the ssnp model and compiles it. 

            Args:
                input_shape: The input and output shape of the ssnp.
                show_summary: If True, the model summary will be printed.
                is_data_normalized (bool, optional): Determine the last layer activation function, sigmoid or relu. Defaults to True (i.e. activation sigmoid).
        """

        output_size = 1
        for i in range(len(input_shape)):
            output_size *= input_shape[i]

        last_decoder_layer_activation_function = 'sigmoid' if is_data_normalized else 'relu'

        encoder = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(seed=SEED),  # type: ignore
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0002)),
            tf.keras.layers.Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(seed=SEED),  # type: ignore
                                  bias_initializer=tf.keras.initializers.Constant(0.01)),  # type: ignore
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(seed=SEED),  # type: ignore
                                  bias_initializer=tf.keras.initializers.Constant(0.01)),  # type: ignore
            tf.keras.layers.Dense(32, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(seed=SEED),  # type: ignore
                                  bias_initializer=tf.keras.initializers.Constant(0.01)),  # type: ignore
            tf.keras.layers.Dense(
                2, activation='sigmoid', bias_initializer=tf.keras.initializers.Constant(0.01)),  # type: ignore
        ], name=ENCODER_NAME)

        decoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(32, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(seed=SEED),  # type: ignore
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0002)),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(seed=SEED),  # type: ignore
                                  bias_initializer=tf.keras.initializers.Constant(0.01)),  # type: ignore
            tf.keras.layers.Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(seed=SEED),  # type: ignore
                                  bias_initializer=tf.keras.initializers.Constant(0.01)),  # type: ignore
            tf.keras.layers.Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(seed=SEED),  # type: ignore
                                  bias_initializer=tf.keras.initializers.Constant(0.01)),  # type: ignore
            tf.keras.layers.Dense(output_size, activation=last_decoder_layer_activation_function,
                                  kernel_initializer=tf.keras.initializers.HeUniform(seed=SEED)),  # type: ignore
            tf.keras.layers.Reshape(input_shape)
        ], name=DECODER_NAME)

        classifier = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax,
                                  kernel_initializer=tf.keras.initializers.HeUniform(seed=SEED))  # type: ignore
        ], name=CLASSIFIER_NAME)

        input_layer = tf.keras.Input(shape=input_shape, name="input")

        encoder_output = encoder(input_layer)
        decoder_output = decoder(encoder_output)
        classifier_output = classifier(decoder_output)

        self.neural_network = tf.keras.models.Model(inputs=input_layer,
                                                    outputs=[decoder_output, classifier_output],
                                                    name=SSNP_NAME)

        self.neural_network.compile(optimizer=tf.keras.optimizers.Adam(),
                                    loss={DECODER_NAME: DECODER_LOSS,
                                          CLASSIFIER_NAME: CLASSIFIER_LOSS},
                                    loss_weights={DECODER_NAME: DECODER_LOSS_WEIGHT,
                                                  CLASSIFIER_NAME: CLASSIFIER_LOSS_WEIGHT},
                                    metrics={DECODER_NAME: "accuracy", CLASSIFIER_NAME: "accuracy"})

        if show_summary:
            self.neural_network.summary(print_fn=self.console.log)

    def fit(self, X: np.ndarray, Y: np.ndarray,
            epochs: int = 100, batch_size: int = 128, 
            is_data_normalized: bool = True):
        """ Fits the model to the specified data.

        Args:
            X (np.ndarray): Train input values
            Y (np.ndarray): Train target values
            epochs (int, optional): The number of epochs. Defaults to 10.
            batch_size (int, optional): Data points used for one batch. Defaults to 128.
            is_data_normalized (bool, optional): Determine the last layer activation function, sigmoid or relu. Defaults to True (i.e. activation sigmoid).
        """
        if self.neural_network is not None:
            self.console.log("Model already loaded. Skipping build.")
            self.encoder = self.neural_network.get_layer(ENCODER_NAME)
            self.decoder = self.neural_network.get_layer(DECODER_NAME)
            return

        self.console.log("Building model according to the data shape.")
        num_classes = len(np.unique(Y))
        self.__build__(input_shape=X.shape[1:], num_classes=num_classes, show_summary=True, is_data_normalized=is_data_normalized)

        stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)
        logger_callback = LoggerModel(name=SSNP_NAME, show_init=False, epochs=epochs, print_fn=self.console.log)

        self.console.log("Fitting model...")

        history = self.neural_network.fit(X, [X, Y],
                                          epochs=epochs,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          validation_split=0.2,
                                          callbacks=[stopping_callback, logger_callback],
                                          verbose=0)

        self.console.log("Model fitted!")
        self.save(history)
        self.encoder = self.neural_network.get_layer(ENCODER_NAME)
        self.decoder = self.neural_network.get_layer(DECODER_NAME)

        # self.show_predictions(dataNd=x_test, labels=y_test)

    def encode(self, data: np.ndarray, verbose: int = 0):
        """ 
        Encodes the data using the encoder part of the autoencoder.

        Args:
            data (np.ndarray): The data to encode.
            verbose (int, optional): Verbosity level. Defaults to 0.

        Returns:
            np.ndarray: The encoded 2D data.
        """
        return self.encoder.predict(data, verbose=verbose)

    def decode(self, data: np.ndarray, verbose: int = 0):
        """ 
        sDecodes the data using the decoder part of the autoencoder.

        Args:
            data (np.ndarray): The data to decode.
            verbose (int, optional): Verbosity level. Defaults to 0.

        Returns:
            np.ndarray: The decoded nD data.
        """
        xNd = self.decoder.predict(data, verbose=verbose)
        return xNd
