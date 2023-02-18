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

import json
import os
import numpy as np
from sklearn.neighbors import KDTree
from math import sqrt

from .Autoencoder import DEFAULT_MODEL_PATH, Autoencoder
from ..DBMInterface import DBMInterface, DBM_DEFAULT_RESOLUTION
from ..tools import get_proj_error

from ...Logger import LoggerInterface

class SDBM(DBMInterface):
    """
        Self Decision Boundary Mapper (SDBM)
        Uses an auto encoder: 
        1) The encoder part of which is used to reduce the dimensionality of the data (nD -> 2D)
        2) The decoder part of which is used to reconstruct the data (2D -> nD)
        3) The classifier is used to predict the class of the data (nD -> 1D) 
        
    """
    
    def __init__(self, classifier, logger:LoggerInterface=None):
        """Initialize an Self Decision Boundary Mapper

        Args:
            classifier (tensorflow.keras.Model): The classifier to be used for the autoencoder
            logger (LoggerInterface, optional): The logger class to be used for outputting the info messages. Defaults to None.
        """
        super().__init__(classifier, logger)
        self.autoencoder = None

    def fit(self, 
            X_train: np.ndarray, Y_train: np.ndarray, 
            X_test: np.ndarray, Y_test: np.ndarray, 
            epochs:int=300, batch_size:int=32,
            load_folder:str = DEFAULT_MODEL_PATH):
        """Train an autoencoder on the training data (this will be used to reduce the dimensionality of the data (nD -> 2D) and decode the 2D space to nD)

        Args:
            X_train (np.ndarray): Training data
            Y_train (np.ndarray): Training labels
            X_test (np.ndarray): Testing data
            Y_test (np.ndarray): Testing labels
            epochs (int, optional): The number of epochs to train the autoencoder. Defaults to 300.
            batch_size (int, optional): Defaults to 32.
            load_folder (str, optional): The folder path which contains a pre-trained autoencoder. Defaults to DEFAULT_MODEL_PATH.

        Returns:
            autoencoder (Autoencoder): The trained autoencoder
        """
        autoencoder = Autoencoder(folder_path = load_folder, classifier=self.classifier)
        autoencoder.fit(X_train, Y_train, X_test, Y_test, epochs, batch_size)        
        return autoencoder
    
    def generate_boundary_map(self, 
                              X_train:np.ndarray, Y_train:np.ndarray,
                              X_test:np.ndarray, Y_test:np.ndarray,
                              train_epochs:int=300, train_batch_size:int=32,
                              resolution:int=DBM_DEFAULT_RESOLUTION,
                              use_fast_decoding:bool=False,
                              projection:str = None # this parameter is not used in SDBM but is placed here to keep the same interface as DBM
                              ):
        """Generate the decision boundary map
        
            Args:
                X_train (np.ndarray): Training data
                Y_train (np.ndarray): Training labels
                X_test (np.ndarray): Testing data
                Y_test (np.ndarray): Testing labels
                train_epochs (int, optional): The number of epochs to train the autoencoder. Defaults to 300.
                train_batch_size (int, optional): Defaults to 32.
                resolution (int, optional): The resolution of the decision boundary map. Defaults to DBM_DEFAULT_RESOLUTION = 256.
                use_fast_decoding (bool, optional): If True, a fast inference algorithm will be used to decode the 2D space and to generate the decision boundary map. Defaults to False.
                projection (str, optional): The projection is not used in SDBM, is placed here just to match the DBM signature. Defaults to None.
                
            Returns:
                img (np.ndarray): The decision boundary map
                img_confidence (np.ndarray): The confidence map
                encoded_training_data (np.ndarray): The 2D coordinates of the training data for each pixel of the decision boundary map
                encdoded_testing_data (np.ndarray): The 2D coordinates of the testing data for each pixel of the decision boundary map
                space_Nd (np.ndarray): The Nd coordinates of the decision boundary map
                history (dict): The history of the training process

            Example:
                >>> import SDBM
                >>> classifier = build_classifier(...)
                >>> sdbm = SDBM.SDBM(classifier)
                >>> img, img_confidence, _, _, space_Nd, history = sdbm.generate_boundary_map(X_train, Y_train, X_test, Y_test)
                >>> plt.imshow(img)
                >>> plt.show()
        """
        
        # first train the autoencoder if it is not already trained
        if self.autoencoder is None:
            self.autoencoder = self.fit(X_train, 
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
        # generate the 2D image in the encoded space
        self.console.log("Decoding the 2D space... 2D -> nD")
        
        save_img_path = os.path.join(DEFAULT_MODEL_PATH, "boundary_map")
        save_img_confidence_path = os.path.join(DEFAULT_MODEL_PATH, "boundary_map_confidence")
        
        if use_fast_decoding:
            img, img_confidence, spaceNd = self._get_img_dbm_fast_(resolution)
            save_img_path += "_fast"
            save_img_confidence_path += "_fast"
        else:
            img, img_confidence, spaceNd = self._get_img_dbm_(resolution)
        
        self.resolution = resolution
        self.spaceNd = spaceNd
        self.X2d = np.concatenate((encoded_training_data, encoded_testing_data), axis=0)
        self.Xnd = np.concatenate((X_train.reshape((X_train.shape[0],-1)), X_test.reshape((X_test.shape[0],-1))), axis=0)
        
        # transform the encoded data to be in the range [0, resolution)
        encoded_testing_data *= (resolution -1)
        encoded_training_data *= (resolution -1)
        encoded_training_data = encoded_training_data.astype(int)
        encoded_testing_data = encoded_testing_data.astype(int)
        
        
        with open(f"{save_img_path}.npy", 'wb') as f:
            np.save(f, img)
        with open(f"{save_img_confidence_path}.npy", 'wb') as f:
            np.save(f, img_confidence)
        
        for [i,j] in encoded_training_data:
            img[i,j] = -1
            img_confidence[i,j] = 1
        for [i,j] in encoded_testing_data:
            img[i,j] = -2
            img_confidence[i,j] = 1
        
        with open(os.path.join(DEFAULT_MODEL_PATH, "history.json"), 'r') as f:
            history = json.load(f)
        
        return (img, img_confidence, encoded_training_data, encoded_testing_data, spaceNd, history)
    
    def _predict2dspace_(self, X2d:np.ndarray):
        """ Predicts the labels for the given 2D data set.

        Args:
            X2d (np.ndarray): The 2D data set
        
        Returns:
            predicted_labels (np.array): The predicted labels for the given 2D data set
            predicted_confidence (np.array): The predicted probabilities for the given 2D data set
            spaceNd (np.array): The decoded nD space
        """
        spaceNd, predictions = self.autoencoder.decode(X2d, verbose=0)
        predicted_labels = np.array([np.argmax(p) for p in predictions])
        predicted_confidence = np.array([np.max(p) for p in predictions])
        return predicted_labels, predicted_confidence, spaceNd
    
    def generate_projection_errors(self, Xnd: np.ndarray = None, X2d: np.ndarray = None, resolution: int = None):
        """ Calculates the projection errors of the given data.

        Args:
            Xnd (np.array): The data to be projected. The data must be in the range [0,1].
            X2d (np.array): The 2D projection of the data. The data must be in the range [0,1].
            resolution (int): The resolution of the 2D space.
        Returns:
            errors (np.array): The projection errors matrix of the given data. 
        
        Example:
            >>> from SDBM import SDBM
            >>> classifier = ...
            >>> Xnd = ...
            >>> sdbm = SDBM(classifier)
            >>> errors = sdbm.get_projection_errors(Xnd)
            >>> plt.imshow(errors)
            >>> plt.show()
        """
        if resolution is None:
            if self.resolution is None:
                self.console.error("The resolution of the 2D space is not set, try to call the method 'generate_boundary_map' first.")
                raise Exception("The resolution of the 2D space is not set, try to call the method 'generate_boundary_map' first.")
            resolution = self.resolution
        if X2d is None:
            if self.X2d is None:
                self.console.error("No 2D data provided and no data stored in the SDBM object.")
                raise ValueError("No 2D data provided and no data stored in the SDBM object.")
            X2d = self.X2d
        if Xnd is None:
            if self.Xnd is None:
                self.console.error("No nD data provided and no data stored in the SDBM object.")
                raise ValueError("No nD data provided and no data stored in the SDBM object.")
            Xnd = self.Xnd
            
        X2d = X2d.reshape((X2d.shape[0], -1))
        Xnd = Xnd.reshape((Xnd.shape[0], -1))
        
        assert len(X2d) == len(Xnd)
        assert X2d.shape[1] == 2
        #resolution = int(sqrt(len(X2d)))
        #assert resolution * resolution == len(X2d)
        
        self.console.log("Calculating the projection errors of the given data")
        errors = np.zeros((resolution,resolution))
        
        K = 10 # Number of nearest neighbors to consider
        metric = "euclidean"
        
        tree = KDTree(X2d, metric=metric)
        self.console.log("Finished computing the 2D tree")
        indices_embedded = tree.query(X2d, k=len(X2d), return_distance=False)
        # Drop the actual point itself
        indices_embedded = indices_embedded[:, 1:]
        self.console.log("Finished computing the 2D tree indices")
        
        
        tree = KDTree(Xnd, metric=metric)
        self.console.log("Finished computing the nD tree")
        indices_source = tree.query(Xnd, k=len(Xnd), return_distance=False)
        # Drop the actual point itself
        indices_source = indices_source[:, 1:]
        self.console.log("Finished computing the nD tree indices")
        
        sparse_map = []
        for k in range(len(X2d)):
            x, y = X2d[k]
            sparse_map.append( (x, y, get_proj_error(indices_source[k], indices_embedded[k], k=K)) )
            
        errors = self._generate_interpolation_rbf_(sparse_map, resolution, function='linear').T
        
        return errors

    