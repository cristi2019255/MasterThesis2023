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

from math import sqrt
import tensorflow as tf
import os
import numpy as np
import json
import matplotlib.pyplot as plt

from ..Logger import Logger, LoggerInterface


class NNinterface:
    """ 
    Defines the interface for the neural networks used by the Decision Boundary Mapper.
    
    Methods to be implemented by the classes that inherit from this class.
    fit: Fits the model to the specified data.
    encode: Encodes the input data. (optional)
    """
    
    def __init__(self, 
                 folder_path: str,                 
                 nn_name: str = "invNN", 
                 logger: LoggerInterface = None):
        """ Initializes the NN interface.

        Args:
            folder_path (str): the folder path where the model will be saved/loaded.
            classifier (tf.keras.model): The classifier model. Defaults to None.
            nn_name (str, optional): Name of the neural network that uses the classifier. Defaults to "invNN".
            logger (LoggerInterface, optional): Defaults to console logging.

        Raises:
            Exception: If the classifier is not provided.
        """
        if logger is not None:
            self.console = logger
        else:
            self.console = Logger(name=nn_name)
                
        self.save_folder_path = folder_path
        self.nn_name = nn_name
        self.neural_network = None
        
        try:
            self.load()
        except Exception as e:
            self.console.error("Error loading the model: {}".format(e))
            self.console.warn("The model will be built and trained from scratch.")
            
    def fit(self):
        """ Fits the model to the specified data."""
        raise NotImplementedError("The fit method is not implemented.")
    
    
    def load(self):
        """
            Loads an auto encoder from the specified folder path. With the .h5 extension.
            Args:
                folder_path (str): The path to the folder where the model is saved.
        """
        try:
            self.neural_network = tf.keras.models.load_model(os.path.join(self.save_folder_path, self.nn_name), compile=False)
            self.console.log("NN loaded successfully")
        except Exception as e:
            self.console.log(f"NN not found. Please check the path folder {self.save_folder_path} and make sure the model is saved there")        
            raise e
    
    def save(self, history = None):
        """
            Saves the model to the specified folder path. With the .tf extension.
        """       
        folder_path = self.save_folder_path 
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        self.neural_network.save(os.path.join(folder_path, self.neural_network.name), save_format="tf")
        self.console.log(f"Model saved to {folder_path}")
        
        if history is None:
            return
        
        with open(os.path.join(folder_path, "history.json"), "w") as f:
            f.write(json.dumps(history.history))
            self.console.log(f"History saved to {folder_path}")
        
        
    def show_predictions(self, dataNd: np.ndarray, labels: np.ndarray, data2d:np.ndarray = None, n_samples:int = 16):
        """ Shows the predictions of the model. By taking first n_samples data points and comparing them to the actual labels.

        Args:
            dataNd (np.ndarray): N-dimensional data points.
            data2d (np.ndarray, optional): 2-dimensional data points. Defaults to None.
            labels (np.ndarray): Actual labels of the data points
            n_samples (int, optional): Number of samples to show. Defaults to 16.
        """
        if data2d is None:
            data2d = self.encode(dataNd)
        
        decoded, predicted = self.decode(data2d)
        predicted_labels = np.argmax(predicted, axis=1)
        
        plt.figure(figsize=(20, 10))
        m = int(sqrt(n_samples))
        n = m * m
        
        for i in range(1,2*n,2):
            plt.subplot(m, 2*m, i)
            plt.imshow(dataNd[i], cmap='gray')
            plt.title(f"Actual: {labels[i]}", color='green' if predicted_labels[i] == labels[i] else 'red', fontsize=12)
            plt.axis('off')
            plt.subplot(m, 2*m, i+1)
            plt.imshow(decoded[i], cmap='gray')
            plt.title(f"Predicted: {predicted_labels[i]}", color='green' if predicted_labels[i] == labels[i] else 'red', fontsize=12)
            plt.axis('off')
        
        plt.show()
    
    def decode(self, data: np.ndarray, verbose: int = 0):
        """ Decodes the data points.

        Args:
            data (np.ndarray): The data points to decode.
            verbose (int, optional): Verbosity mode. Defaults to 0.
            
        Returns:
            Xnd, labels_predictions: The decoded data points and the predictions of the classifier.
        """
        return self.neural_network.predict(data, verbose=verbose)
    
    def encode(self, data: np.ndarray, verbose: int = 0):
        """ Encodes the data points.

        Args:
            data (np.ndarray): The data points to encode.
            verbose (int, optional): Verbosity mode. Defaults to 0.

        Returns:
            np.ndarray: The encoded 2D data points.
        """
        return NotImplementedError("The encode method is not implemented.")