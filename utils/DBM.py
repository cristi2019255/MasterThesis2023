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
import tensorflow as tf
from models.autoencoder import build_autoencoder, load_autoencoder
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils.Logger import Logger


DBM_DEFAULT_RESOLUTION = 100

class DBM:
    """
        DBM - Decision Boundary Mapper
    """
    def __init__(self, classifier):
        self.console = Logger(name="Decision Boundary Mapper - DBM, using Autoencoder")
        self.console.log("Loaded classifier: " + classifier.name + "")
        self.classifier = classifier
        self.autoencoder = None

    def fit(self, X_train, Y_train, X_test, Y_test, epochs=10, batch_size=128, show_predictions=True, load_autoencoder_folder = os.path.join("models", "model", "DBM", "MNIST")):
        num_classes = np.unique(Y_train).shape[0]
        data_shape = X_train.shape[1:]
        
        # 1. Train an autoencoder on the training data (this will be used to reduce the dimensionality of the data) nD -> 2D
        try:
            autoencoder = load_autoencoder(load_autoencoder_folder)
            self.console.log("Loaded autoencoder from disk")
        except Exception as e:
            self.console.log("Could not load autoencoder from disk. Training a new one.")
            autoencoder = build_autoencoder(self.classifier, data_shape, num_classes, show_summary=True)
            autoencoder.fit(X_train, Y_train, X_test, Y_test, epochs=epochs, batch_size=batch_size)
        
        if show_predictions:
            autoencoder.show_predictions(X_test[:20], Y_test[:20])

        self.autoencoder = autoencoder
        self.classifier = autoencoder.classifier
        return autoencoder
    
    def generate_boundary_map(self, 
                              X_train, Y_train, X_test, Y_test,
                              train_epochs=10, train_batch_size=128,
                              show_autoencoder_predictions=True,
                              show_encoded_corpus=True,
                              resolution=DBM_DEFAULT_RESOLUTION, 
                              show_mapping=True, 
                              save_file_path=os.path.join("results", "MNIST", "2D_boundary_mapping.png"),
                              class_name_mapper = lambda x: str(x)
                              ):
        
        # first train the autoencoder if it is not already trained
        if self.autoencoder is None:
            self.fit(X_train, 
                    Y_train, 
                    X_test, 
                    Y_test, 
                    epochs=train_epochs, 
                    batch_size=train_batch_size, 
                    show_predictions=show_autoencoder_predictions)

        # encoder the train and test data and show the encoded data in 2D space
        encoded_training_data = self.autoencoder.encode(X_train)
        encoded_testing_data = self.autoencoder.encode(X_test)            
        
        if show_encoded_corpus:
            plt.figure(figsize=(20, 20))
            plt.title("Encoded data in 2D space")
            plt.axis('off')
            plt.plot(encoded_training_data[:,0], encoded_training_data[:,1], 'ro', label="Training data", alpha=0.5)
            plt.plot(encoded_testing_data[:,0], encoded_testing_data[:,1], 'bo', label="Testing data", alpha=0.5)
            plt.show()
            
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
        spaceNd = self.autoencoder.decode(space)
        predictions = self.classifier.predict(spaceNd)
        predicted_labels = np.array([np.argmax(p) for p in predictions])
        img = predicted_labels.reshape((resolution, resolution))
        #self.console.log("2D boundary mapping: \n", img)

        self._build_2D_image(encoded_training_data, encoded_testing_data, img, show_mapping, save_file_path, class_name_mapper)
    
        return img

    def _build_2D_image(self, encoded_training_data, encoded_testing_data, img, show_mapping, save_file_path, class_name_mapper = lambda x: str(x)):
        color_img = np.zeros((img.shape[0], img.shape[1], 3))
    
        for [i,j] in encoded_training_data:
            img[i,j] = -1
        for [i,j] in encoded_testing_data:
            img[i,j] = -2
        
        
        with open(f"{save_file_path}.npy", 'wb') as f:
            np.save(f, img)
        
        colors_mapper = {
            -2: [0,0,0], # setting original test data to black
            -1: [1,1,1], # setting original train data to white
            0: [1,0,0], 
            1: [0,1,0], 
            2: [0,0,1], 
            3: [1,1,0], 
            4: [0,1,1], 
            5: [1,0,1], 
            6: [0.5,0.5,0.5], 
            7: [0.5,0,0], 
            8: [0,0.5,0], 
            9: [0,0,0.5]
        }
        
        for i, j in np.ndindex(img.shape):
            color_img[i,j] = colors_mapper[img[i,j]]
        
        plt.figure(figsize=(13, 10))
        plt.title("2D boundary mapping")
        plt.axis('off')
        
        values = np.unique(img)
        im =  plt.imshow(color_img)
        
        patches = []
        for value in values:
            color = colors_mapper[value]
            if value==-1:
                label = "Original train data"
            elif value==-2:
                label = "Original test data"
            else:
                label = f"Value region: {class_name_mapper(value)}"

            patches.append(mpatches.Patch(color=color, label=label))
        
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
        
        plt.savefig(save_file_path)
        
        if show_mapping:
            plt.show()
        
        self.console.success(f"2D boundary mapping saved to: {save_file_path}")