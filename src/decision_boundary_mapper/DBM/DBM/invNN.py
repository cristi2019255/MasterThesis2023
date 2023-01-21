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
import matplotlib.pyplot as plt

from ...Logger import Logger, LoggerInterface

DEFAULT_MODEL_PATH = os.path.join("models", "DBM")

class invNN:
    """
        Inverse Projection Neural Network.
        The network is composed of 2 parts:
            1) The decoder part, i.e. The inverse projection 2D -> nD. (Sequential model 2 -> 32 -> 128 -> 512 -> nD) if is not provided by the user
            2) The classifier part nD -> 1D (provided by the user).
    """
    
    def __init__(self, 
                 decoder = None, 
                 classifier = None, 
                 logger: LoggerInterface = None, 
                 load: bool = False,
                 folder_path: str = DEFAULT_MODEL_PATH):
        """
            Creates an inverse Projection Neural Network model.
            Decoder: The decoder part, i.e. The inverse projection 2D -> nD.
            Classifier: The classifier part nD -> 1D.
        """
        if logger is not None:
            self.console = logger
        else:
            self.console = Logger(name="inverse Projection Neural Network")
        
        self.save_folder_path = folder_path
        self.decoder = decoder
        self.classifier = classifier
        
        if load:
            self.load(folder_path)
            
        self.pre_load = load
        
    def load(self, folder_path:str):
        """
            Loads an auto encoder from the specified folder path. With the .h5 extension.
            Args:
                folder_path (str): The path to the folder where the model is saved.
        """
        self.save_folder_path = folder_path
        try:
            self.classifier = tf.keras.models.load_model(os.path.join(folder_path, "classifier.h5"), compile=False)
            self.decoder = tf.keras.models.load_model(os.path.join(folder_path, "decoder.h5"), compile=False)
            self.decoder_classifier = tf.keras.models.load_model(os.path.join(folder_path, "decoder_classifier.h5"), compile=False)
            self.invNN = tf.keras.models.load_model(os.path.join(folder_path, "invNN.h5"), compile=False)
        except Exception as e:
            self.console.log(f"Inverse projection NN not found. Please check the path folder {folder_path} and make sure the model is saved there")        
            self.console.error(f"Exception: {e}")
            raise e
    
    def __build__(self, output_shape:tuple = (2,1)):
        """Builds an invNN if not provided by the user (Sequential 2D -> 32 -> 128 -> 512 -> nD -> ... -> 1D)

        Args:
            output_shape (tuple, optional): _description_. Defaults to (2,1).
        """
        
        assert len(output_shape) == 2, "Output shape must be a 2D tuple"
        
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(output_shape[0] * output_shape[1], activation='linear'),
            tf.keras.layers.Reshape(output_shape)
        ], name="decoder")

        self.decoder_classifier = tf.keras.Sequential([
            self.decoder,
            self.classifier
        ], name="decoder_classifier")  
        
        input_layer = tf.keras.Input(shape=(2,), name="input")
                                             
        decoder_classifier_output = self.decoder_classifier(input_layer)
        decoder_output = self.decoder(input_layer)
            
        self.invNN = tf.keras.models.Model(inputs=input_layer, 
                                            outputs=[decoder_output, 
                                                    decoder_classifier_output], 
                                            name="invNN")
        
        optimizer = tf.keras.optimizers.Adam()
        
        self.invNN.compile(optimizer=optimizer, 
                                loss={"decoder":"binary_crossentropy",
                                      "decoder_classifier": "sparse_categorical_crossentropy"}, 
                                metrics=['accuracy'])
                                             
    def summary(self):
        self.invNN.summary()
        
    def fit(self, 
            x2d_train: np.ndarray, xNd_train: np.ndarray, y_train: np.ndarray, 
            x2d_test: np.ndarray, xNd_test: np.ndarray, y_test: np.ndarray, 
            epochs:int = 10, batch_size:int = 128):
        """ Fits the model to the specified data.

        Args:
            x2d_train (np.ndarray): Train input values (2D)
            xNd_train (np.ndarray): Train input values (nD)
            y_train (np.ndarray): Train target values
            x2d_test (np.ndarray): Test input values (2D)
            xNd_test (np.ndarray): Test input values (nD)
            y_test (np.ndarray): Test target values
            epochs (int, optional): The number of epochs. Defaults to 10.
            batch_size (int, optional): Data points used for one batch. Defaults to 128.
        """
        if self.pre_load:
            self.console.log("Model already loaded. Skipping build.")
        else:
            self.console.log("Building model according to the data shape.")
            self.__build__(xNd_train.shape[1:])
            
        self.console.log("Fitting model...")
        self.invNN.fit(x2d_train, [xNd_train, y_train], 
                            epochs=epochs, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            validation_data=(x2d_test, [xNd_test, y_test]))
        self.console.log("Model fitted!")
        self.save()
        
    def save(self):
        """
            Saves the model to the specified folder path. With the .h5 extension.
        """       
        folder_path = self.save_folder_path 
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        self.classifier.save(os.path.join(folder_path, "classifier.h5"))
        self.decoder.save(os.path.join(folder_path, "decoder.h5"))
        self.decoder_classifier.save(os.path.join(folder_path, "decoder_classifier.h5"))
        self.invNN.save(os.path.join(folder_path, "invNN.h5"))
        self.console.log(f"Model saved to {folder_path}")
        
    def show_predictions(self, data: np.ndarray, labels: np.ndarray):
        """ Shows the predictions of the model. By taking first 20 data points and comparing them to the actual labels.

        Args:
            data (np.ndarray): Data set
            labels (np.ndarray): Actual labels of the data points
        """
        decoded = self.decoder.predict(data)
        decoded_labels = self.decoder_classifier.predict(data)
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
    
    def decode(self, data: np.ndarray):
        return self.decoder.predict(data, verbose=0)
    