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
import numpy as np
from DBM.DBMInterface import DBMInterface
from DBM.DBMInterface import DBM_DEFAULT_RESOLUTION
from models.Autoencoder import build_autoencoder, load_autoencoder
import matplotlib.pyplot as plt

from utils.Logger import Logger



class SDBM(DBMInterface):
    """
        SDBM - Self Decision Boundary Mapper
    """
    def __init__(self, classifier, logger=None):
        super().__init__(classifier, logger)
        self.autoencoder = None

    def fit(self, 
            X_train, 
            Y_train, 
            X_test, 
            Y_test, 
            epochs=10,
            batch_size=128,
            load_folder = os.path.join("models", "model", "DBM", "MNIST")):
        
        num_classes = np.unique(Y_train).shape[0]
        data_shape = X_train.shape[1:]
        
        # 1. Train an autoencoder on the training data (this will be used to reduce the dimensionality of the data) nD -> 2D
        try:
            autoencoder = load_autoencoder(load_folder)
            self.console.log("Loaded autoencoder from disk")
        except Exception as e:
            self.console.log("Could not load autoencoder from disk. Training a new one.")
            autoencoder = build_autoencoder(self.classifier, data_shape, num_classes, show_summary=True)
            autoencoder.fit(X_train, Y_train, X_test, Y_test, epochs=epochs, batch_size=batch_size)
        
        #if show_predictions:
        #    autoencoder.show_predictions(X_test[:20], Y_test[:20])

        self.autoencoder = autoencoder
        self.classifier = autoencoder.classifier
        return autoencoder
    
    def generate_boundary_map(self, 
                              X_train, Y_train, X_test, Y_test,
                              train_epochs=10, 
                              train_batch_size=128,
                              show_encoded_corpus=True,
                              resolution=DBM_DEFAULT_RESOLUTION, 
                              ):
        
        # making sure that the data is of the correct type
        assert type(X_train) == np.ndarray
        assert type(Y_train) == np.ndarray
        assert type(X_test) == np.ndarray
        assert type(Y_test) == np.ndarray
        
        
        # first train the autoencoder if it is not already trained
        if self.autoencoder is None:
            self.fit(X_train, 
                    Y_train, 
                    X_test, 
                    Y_test, 
                    epochs=train_epochs, 
                    batch_size=train_batch_size)

        # encoder the train and test data and show the encoded data in 2D space
        self.console.log("Encoding the training data to 2D space")
        encoded_training_data = self.autoencoder.encode(X_train)
        self.console.log("Encoding the testing data to 2D space")
        encoded_testing_data = self.autoencoder.encode(X_test)            
        
        # Skip the visualization of the encoded data
        #if show_encoded_corpus:
        #    plt.figure(figsize=(20, 20))
        #    plt.title("Encoded data in 2D space")
        #    plt.axis('off')
        #    plt.plot(encoded_training_data[:,0], encoded_training_data[:,1], 'ro', label="Training data", alpha=0.5)
        #    plt.plot(encoded_testing_data[:,0], encoded_testing_data[:,1], 'bo', label="Testing data", alpha=0.5)
        #    plt.show()
            
        # getting the max and min values for the encoded data
        min_x = min(np.min(encoded_training_data[:,0]), np.min(encoded_testing_data[:,0]))
        max_x = max(np.max(encoded_training_data[:,0]), np.max(encoded_testing_data[:,0]))
        min_y = min(np.min(encoded_training_data[:,1]), np.min(encoded_testing_data[:,1]))
        max_y = max(np.max(encoded_training_data[:,1]), np.max(encoded_testing_data[:,1]))
        
        # transform the encoded data to be in the range [0, resolution]
        encoded_training_data[:,0] = (encoded_training_data[:,0] - min_x) / (max_x - min_x) * (resolution - 1)
        encoded_testing_data[:,0] = (encoded_testing_data[:,0] - min_x) / (max_x - min_x) * (resolution - 1)
        encoded_training_data[:,1] = (encoded_training_data[:,1] - min_y) / (max_y - min_y) * (resolution - 1)
        encoded_testing_data[:,1] = (encoded_testing_data[:,1] - min_y) / (max_y - min_y) * (resolution - 1)
        
        encoded_training_data = encoded_training_data.astype(int)
        encoded_testing_data = encoded_testing_data.astype(int)
        
        # generate the 2D image in the encoded space
        space = np.array([(i / resolution * (max_x - min_x) + min_x, j / resolution * (max_y - min_y) + min_y) for i in range(resolution) for j in range(resolution)])
        self.console.log("Decoding the 2D space... 2D -> nD")
        spaceNd = self.autoencoder.decode(space)
        self.console.log("Predicting labels for the 2D boundary mapping using the nD data and the trained classifier...")
        predictions = self.classifier.predict(spaceNd)
        predicted_labels = np.array([np.argmax(p) for p in predictions])
        img = predicted_labels.reshape((resolution, resolution))
        #self.console.log("2D boundary mapping: \n", img)

        for [i,j] in encoded_training_data:
            img[i,j] = -1
        for [i,j] in encoded_testing_data:
            img[i,j] = -2
        
        #with open(f"{save_file_path}.npy", 'wb') as f:
        #    np.save(f, img)
        

        return (img, encoded_training_data, encoded_testing_data)