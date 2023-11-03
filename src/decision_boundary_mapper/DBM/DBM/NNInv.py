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
import os

from ..AbstractNN import AbstractNN, SEED
from ...Logger import LoggerInterface, LoggerModel

DEFAULT_MODEL_PATH = os.path.join("tmp", "DBM")
NNINV_NAME = "NNInv"
DECODER_NAME = "decoder"
DECODER_LOSS = "mean_squared_error"


class NNInv(AbstractNN):
    """
        Inverse Projection Neural Network.        
        The inverse projection 2D -> nD. 
        (Sequential model 2 -> 32 -> 64 -> 128 -> 512 -> nD)            
    """

    def __init__(self,
                 logger: LoggerInterface | None = None,
                 folder_path: str = DEFAULT_MODEL_PATH):
        """
            Creates an inverse Projection Neural Network model.
        """
        super().__init__(folder_path=folder_path, nn_name=NNINV_NAME, logger=logger)

    def __build__(self, output_shape: tuple = (2, 2), show_summary: bool = False, is_data_normalized: bool = True):
        """
        Builds an NNInv model
        (Sequential 2D -> 32 -> 64 -> 128 -> 512 -> nD)

        Args:
            output_shape (tuple, optional): The output shape of the Nd data. Defaults to (2,2).
            show_summary (bool, optional): If True, the model summary will be printed. Defaults to False.
            is_data_normalized (bool, optional): Determine the last layer activation function, sigmoid or relu. Defaults to True (i.e. activation sigmoid).
        """

        # computing the output size
        output_size = 1
        for i in range(len(output_shape)):
            output_size *= output_shape[i]

        last_layer_activation_function = 'sigmoid' if is_data_normalized else 'relu'

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(seed=SEED),  # type: ignore
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0002)),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(seed=SEED),  # type: ignore
                                  bias_initializer=tf.keras.initializers.Constant(0.01)),  # type: ignore
            tf.keras.layers.Dense(128, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(seed=SEED),  # type: ignore
                                  bias_initializer=tf.keras.initializers.Constant(0.01)),  # type: ignore
            tf.keras.layers.Dense(512, activation='relu', kernel_initializer=tf.keras.initializers.HeUniform(seed=SEED),  # type: ignore
                                  bias_initializer=tf.keras.initializers.Constant(0.01)),  # type: ignore
            tf.keras.layers.Dense(output_size, activation=last_layer_activation_function,
                                  kernel_initializer=tf.keras.initializers.HeUniform(seed=SEED)),  # type: ignore
            tf.keras.layers.Reshape(output_shape)
        ], name=DECODER_NAME)

        input_layer = tf.keras.Input(shape=(2,), name="input")

        self.neural_network = tf.keras.models.Model(inputs=input_layer,
                                                    outputs=[self.decoder(input_layer)],
                                                    name=NNINV_NAME)

        self.neural_network.compile(optimizer=tf.keras.optimizers.Adam(),
                                    loss=DECODER_LOSS,
                                    metrics=["accuracy"])

        if show_summary:
            self.neural_network.summary(print_fn=self.console.log)

    def fit(self,
            x2d: np.ndarray, xNd: np.ndarray,
            epochs: int = 300, batch_size: int = 32,
            is_data_normalized: bool = True,
            ):
        """ 
        Fits the model to the specified data.

        Args:
            x2d (np.ndarray): Train input values (2D)
            xNd (np.ndarray): Train input values (nD)
            epochs (int, optional): The number of epochs. Defaults to 300.
            batch_size (int, optional): Data points used for one batch. Defaults to 32.
            is_data_normalized (bool, optional): Determine the last layer activation function, sigmoid or relu. Defaults to True (i.e. activation sigmoid).
        """
        if self.neural_network is not None:
            self.console.log("Model already loaded. Skipping build.")
            return

        self.console.log("Building model according to the data shape.")
        self.__build__(output_shape=xNd.shape[1:], show_summary=True, is_data_normalized=is_data_normalized)

        stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)
        logger_callback = LoggerModel(name=NNINV_NAME, show_init=False, epochs=epochs, print_fn=self.console.log)
        self.console.log("Fitting model...")

        hist = self.neural_network.fit(x2d, xNd,
                                       epochs=epochs,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       validation_split=0.2,
                                       callbacks=[stopping_callback,
                                                  logger_callback],
                                       verbose=0)

        self.console.log("Model fitted!")
        self.save(hist)
        # self.show_predictions(dataNd=xNd_test, data2d=x2d_test, labels=y_test)
