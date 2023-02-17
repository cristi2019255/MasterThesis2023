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

import os
import tensorflow as tf
import numpy as np

from ..NNinterface import NNinterface
from ...Logger import LoggerInterface

DEFAULT_MODEL_PATH = os.path.join("tmp", "SDBM")
DECODER_NAME = "decoder"
ENCODER_NAME = "encoder"
AUTOENCODER_NAME = "autoencoder"

class Autoencoder(NNinterface):
    def __init__(self, 
                 classifier = None, 
                 folder_path:str = DEFAULT_MODEL_PATH, 
                 logger:LoggerInterface = None):
        """
            Creates an autoencoder model.
            Classifier: The classifier part of the autoencoder.
            
            Args:
                folder_path: The path to the folder where the model will be saved/loaded.
                logger: The logger used to log the model's progress.
                classifier: The classifier part of the autoencoder.
        """
        super().__init__(folder_path=folder_path, logger=logger, nn_name=AUTOENCODER_NAME, classifier=classifier)
    
    
    def __build__(self, input_shape:tuple = (28, 28), show_summary:bool = False):
        """
            Assembles the autoencoder model and compiles it. 
            
            Args:
                input_shape: The input and output shape of the autoencoder.
                show_summary: If True, the model summary will be printed.
        """
        
        encoder = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.0002)),
            tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform', bias_initializer=tf.keras.initializers.Constant(0.01)),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform', bias_initializer=tf.keras.initializers.Constant(0.01)),
            tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform', bias_initializer=tf.keras.initializers.Constant(0.01)),
            tf.keras.layers.Dense(2, activation='sigmoid', bias_initializer=tf.keras.initializers.Constant(0.01)),            
        ], name=ENCODER_NAME)

        decoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.0002)),
            tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform', bias_initializer=tf.keras.initializers.Constant(0.01)),
            tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform', bias_initializer=tf.keras.initializers.Constant(0.01)),
            tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform', bias_initializer=tf.keras.initializers.Constant(0.01)),
            tf.keras.layers.Dense(input_shape[0] * input_shape[1], activation='sigmoid'),
            tf.keras.layers.Reshape(input_shape)
        ], name=DECODER_NAME)       
        
        
        input_layer = tf.keras.Input(shape=input_shape, name="input")
        
        CLASSIFIER_NAME = self.classifier.name
        DECODER_LOSS = "mean_squared_error"
        CLASSIFIER_LOSS = "sparse_categorical_crossentropy"
        DECODER_LOSS_WEIGHT = 1.0
        CLASSIFIER_LOSS_WEIGHT = 0.125
        
        encoder_output = encoder(input_layer)
        decoder_output = decoder(encoder_output)
        classifier_output = self.classifier(decoder_output)
        
        self.neural_network = tf.keras.models.Model(inputs=input_layer,
                                                    outputs=[decoder_output, classifier_output],
                                                    name=AUTOENCODER_NAME)
        
        self.neural_network.compile(optimizer=tf.keras.optimizers.Adam(), 
                                    loss = {DECODER_NAME:DECODER_LOSS,
                                            CLASSIFIER_NAME:CLASSIFIER_LOSS},
                                    loss_weights = {DECODER_NAME:DECODER_LOSS_WEIGHT,
                                                    CLASSIFIER_NAME:CLASSIFIER_LOSS_WEIGHT},
                                    metrics = {DECODER_NAME: "accuracy", CLASSIFIER_NAME: "accuracy"})
        
        if show_summary:
            self.neural_network.summary()
        
        
    def fit(self, x_train: np.ndarray, y_train: np.ndarray, 
            x_test: np.ndarray, y_test: np.ndarray,
            epochs:int = 10, batch_size:int = 128):
        """ Fits the model to the specified data.

        Args:
            x_train (np.ndarray): Train input values
            y_train (np.ndarray): Train target values
            x_test (np.ndarray): Test input values
            y_test (np.ndarray): Test target values
            epochs (int, optional): The number of epochs. Defaults to 10.
            batch_size (int, optional): Data points used for one batch. Defaults to 128.
        """
        if self.neural_network is not None:
            self.console.log("Model already loaded. Skipping build.")
            self.encoder = self.neural_network.get_layer(ENCODER_NAME)
            self.decoder = self.neural_network.get_layer(DECODER_NAME)
            self.classifier = self.neural_network.get_layer(self.classifier.name)
            return
        
        self.console.log("Building model according to the data shape.")
        self.__build__(input_shape=x_train.shape[1:], show_summary=True)
            
        stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.00001, mode='min', patience=20, restore_best_weights=True)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(self.save_folder_path, "checkpoint.cpkt"),
                                                             save_weights_only=True,
                                                             verbose=1)

        self.console.log("Fitting model...")
        
        history = self.neural_network.fit(x_train, [x_train, y_train], 
                            epochs=epochs, 
                            batch_size=batch_size, 
                            shuffle=True,
                            validation_data=(x_test, [x_test, y_test]),
                            callbacks=[stopping_callback, checkpoint_callback])
        
        self.console.log("Model fitted!")
        self.save(history)    
        self.encoder = self.neural_network.get_layer(ENCODER_NAME)
        self.decoder = self.neural_network.get_layer(DECODER_NAME)
        self.classifier = self.neural_network.get_layer(self.classifier.name)
            
        self.show_predictions(dataNd=x_test, labels=y_test)
            
    def encode(self, data:np.ndarray, verbose:int = 0):
        """ Encodes the data using the encoder part of the autoencoder.

        Args:
            data (np.ndarray): The data to encode.
            verbose (int, optional): Verbosity level. Defaults to 0.

        Returns:
            np.ndarray: The encoded 2D data.
        """
        return self.encoder.predict(data, verbose=verbose)
        
    def decode(self, data:np.ndarray, verbose:int = 0):
        """ Decodes the data using the decoder part of the autoencoder.

        Args:
            data (np.ndarray): The data to decode.
            verbose (int, optional): Verbosity level. Defaults to 0.

        Returns:
            np.ndarray: The decoded nD data and classifier predictions.
        """
        xNd = self.decoder.predict(data, verbose=verbose)
        predictions = self.classifier.predict(xNd, verbose=verbose)
        return xNd, predictions
    