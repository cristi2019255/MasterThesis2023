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
import matplotlib.pyplot as plt
import numpy as np

from ...Logger import Logger, LoggerInterface

DEFAULT_MODEL_PATH = os.path.join("models", "SDBM")

def build_autoencoder(classifier, input_shape:tuple = (28, 28), show_summary:bool = False):
    """ Building an autoencoder for dimensionality reduction of the data

    Args:
        classifier (tensorflow.keras.Model): The classifier to be used for the autoencoder
        input_shape (tuple, optional): Data input shape. Defaults to (28, 28).
        num_classes (int, optional): Target number of classes to be predicted. Defaults to 10.
        show_summary (bool, optional): If true shows the autoencoder summary. Defaults to False.

    Returns:
        autoencoder (tensorflow.keras.Model): The autoencoder model
    """
    ENCODER = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu', bias_initializer=tf.keras.initializers.Constant(10e-4)),
        tf.keras.layers.Dense(128, activation='relu', bias_initializer=tf.keras.initializers.Constant(10e-4)),
        tf.keras.layers.Dense(32, activation='relu', bias_initializer=tf.keras.initializers.Constant(10e-4)),
        tf.keras.layers.Dense(2, activation='linear', activity_regularizer=tf.keras.regularizers.l2(1/2), bias_initializer=tf.keras.initializers.Constant(10e-4)),
    ], name="encoder")

    DECODER = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, activation='relu', bias_initializer=tf.keras.initializers.Constant(10e-4)),
        tf.keras.layers.Dense(128, activation='relu', bias_initializer=tf.keras.initializers.Constant(10e-4)),
        tf.keras.layers.Dense(512, activation='relu', bias_initializer=tf.keras.initializers.Constant(10e-4)),
        tf.keras.layers.Dense(input_shape[0] * input_shape[1], activation='sigmoid'),
        tf.keras.layers.Reshape(input_shape)
    ], name="decoder")       
    
    
    INPUT = tf.keras.Input(shape=input_shape, name="input")
    
    autoencoder = Autoencoder(ENCODER, DECODER, classifier, INPUT)
    
    if show_summary:
        autoencoder.summary()
        
    return autoencoder

class Autoencoder:
    def __init__(self, encoder = None, decoder = None, classifier = None, input_layer = None, 
                 folder_path:str = DEFAULT_MODEL_PATH, load:bool = False, logger:LoggerInterface = None):
        """
            Creates an autoencoder model.
            Encoder: The encoder part of the autoencoder.
            Decoder: The decoder part of the autoencoder.
            Classifier: The classifier part of the autoencoder.
            Input_layer: The input layer of the autoencoder.
        """
        if logger is not None:
            self.console = logger
        else:
            self.console = Logger(name="Autoencoder")
        
        self.save_folder_path = folder_path
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.input_layer = input_layer
        
        if load:
            self.load(folder_path)
        else:
            self.build()
    
    def load(self, folder_path:str):
        """
            Loads an autoencoder from the specified folder path. With the .h5 extension.
            Args:
                folder_path (str): The folder path where the autoencoder is saved.
        """
        self.save_folder_path = folder_path
        try:
            self.autoencoder_classifier = tf.keras.models.load_model(os.path.join(folder_path, "autoencoder_classifier.h5"), compile=False)
            self.auto_encoder = tf.keras.models.load_model(os.path.join(folder_path, "auto_encoder.h5"), compile=False)
            self.classifier = tf.keras.models.load_model(os.path.join(folder_path, "classifier.h5"), compile=False)
            self.autoencoder = tf.keras.models.load_model(os.path.join(folder_path, "autoencoder.h5"), compile=False)
            self.decoder = tf.keras.models.load_model(os.path.join(folder_path, "decoder.h5"), compile=False)
            self.encoder = tf.keras.models.load_model(os.path.join(folder_path, "encoder.h5"), compile=False)
        except Exception as e:
            self.console.log("Autoencoder not found. Please check the path folder and make sure the autoencoder is saved there")        
            self.console.error(f"Exception: {e}")
            raise e
    
    
    def build(self):
        """Assembles the autoencoder model and compiles it. 
        """
        self.autoencoder_classifier = tf.keras.models.Sequential([
            self.encoder,
            self.decoder,
            self.classifier,
        ], name="autoencoder_classifier")
    
        self.auto_encoder = tf.keras.models.Sequential([
            self.encoder,
            self.decoder,
        ], name="auto_encoder")
        
        auto_encoder_classifier_output = self.autoencoder_classifier(self.input_layer)
        auto_encoder_output = self.auto_encoder(self.input_layer)
        
            
        self.autoencoder = tf.keras.models.Model(inputs=self.input_layer, 
                                            outputs=[auto_encoder_output, 
                                                    auto_encoder_classifier_output], 
                                            name="autoencoder")
        
        optimizer = tf.keras.optimizers.Adam()
        
        self.autoencoder.compile(optimizer=optimizer, 
                                loss={"auto_encoder":"binary_crossentropy",
                                      "autoencoder_classifier": "sparse_categorical_crossentropy"}, 
                                metrics=['accuracy'])
    
    def summary(self):
        self.autoencoder.summary()
        
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
        self.autoencoder.fit(x_train, [x_train, y_train], 
                            epochs=epochs, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            validation_data=(x_test, [x_test, y_test]))
        
        self.save()
        
    def save(self):
        """
            Saves the model to the specified folder path. With the .h5 extension.
        """       
        folder_path = self.save_folder_path 
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        self.auto_encoder.save(os.path.join(folder_path, "auto_encoder.h5"))
        self.autoencoder_classifier.save(os.path.join(folder_path, "autoencoder_classifier.h5"))
        self.autoencoder.save(os.path.join(folder_path, "autoencoder.h5"))
        self.decoder.save(os.path.join(folder_path, "decoder.h5"))
        self.encoder.save(os.path.join(folder_path, "encoder.h5"))
        self.classifier.save(os.path.join(folder_path, "classifier.h5"))
    
    def show_predictions(self, data:np.ndarray, labels:np.ndarray):
        """Shows the predictions of the autoencoder. First 20 data points from the provided data input.

        Args:
            data (np.ndarray): The data set to be predicted.
            labels (np.ndarray): The actual labels of the data set.
        """
        decoded = self.auto_encoder.predict(data)
        decoded_labels = self.autoencoder_classifier.predict(data)
        predicted_labels = [np.argmax(label) for label in decoded_labels]
        
        plt.figure(figsize=(20, 6))
        
        for i in range(1,20,2):
            plt.subplot(2, 10, i)
            plt.imshow(data[i], cmap='gray')
            plt.axis('off')
            plt.subplot(2, 10, i+1)
            plt.imshow(decoded[i], cmap='gray')
            plt.title(f"Predicted: {predicted_labels[i]} \n Actual: {labels[i]}", color='green' if predicted_labels[i] == labels[i] else 'red')
            plt.axis('off')
        
        plt.show()
    
    def encode(self, data:np.ndarray):
        #self.console.log("Encoding data")
        return self.encoder.predict(data)
    
    def decode(self, data:np.ndarray):
        #self.console.log("Decoding data")
        return self.decoder.predict(data, verbose=0)
    