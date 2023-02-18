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

from .invNN import DEFAULT_MODEL_PATH, invNN
from .projections import PROJECTION_METHODS


from ..DBMInterface import DBMInterface, DBM_DEFAULT_RESOLUTION
from ..tools import get_proj_error

from ...Logger import LoggerInterface

class DBM(DBMInterface):
    """
    The Decision Boundary Map (DBM) class has methods for visualizing the decision boundaries of a classifier.

    Methods:
        fit: Trains the classifier on the given data set.
        get_decision_boundary_map: Returns the decision boundary map for the given classifier.
        
    Example:
        >>> from DBM import DBM
        >>> classifier = ...
        >>> dbm = DBM(classifier)
        >>> img, img_confidence, img_projection_errors, img_inverse_projection_errors, X2d_train, X2d_test = dbm.get_decision_boundary_map(X2d_train, Xnd_train, Y_train, X2d_test, Xnd_test, Y_test)
        >>> plt.imshow(img)
        >>> plt.show()
    """
    
    def __init__(self, classifier, logger:LoggerInterface=None):
        """ Initializes the DBM class.

        Args:
            classifier (tensorflow.keras.Model): The classifier to be used for the decision boundary map.
            logger (LoggerInterface, optional): The logger for outputting info messages. Defaults to console logging.
        """
        super().__init__(classifier, logger)
        self.invNN = None
        
        # creating a folder for the model
        if not os.path.exists(DEFAULT_MODEL_PATH):
           os.makedirs(DEFAULT_MODEL_PATH) 
                                                  
    def fit(self, 
            X2d_train: np.ndarray, Xnd_train: np.ndarray, Y_train: np.ndarray,
            X2d_test: np.ndarray, Xnd_test: np.ndarray, Y_test: np.ndarray, 
            epochs:int=300, batch_size:int=32, 
            load_folder:str=DEFAULT_MODEL_PATH):
        """ 
        Trains the classifier on the given data set.

        Args:
            X2d_train (np.ndarray): Training data set 2D data got from the projection of the original data (e.g. PCA, t-SNE, UMAP)
            Xnd_train (np.ndarray): Training data set nD data (e.g. MNIST, CIFAR10) (i.e. the original data)
            Y_train (np.ndarray): Training data set labels
            X2d_test (np.ndarray): Testing data set 2D data got from the projection of the original data (e.g. PCA, t-SNE, UMAP)
            Xnd_test (np.ndarray): Testing data set nD data (e.g. MNIST, CIFAR10) (i.e. the original data)
            Y_test (np.ndarray): Testing data set nD data (e.g. MNIST, CIFAR10) (i.e. the original data)
            epochs (int, optional): The number of epochs for which the DBM is trained. Defaults to 300.
            batch_size (int, optional): Train batch size. Defaults to 32.
        
        Returns:
            inverse_porjection_NN (invNN): The trained inverse projection neural network.
        """
        
        inverse_projection_NN = invNN(classifier=self.classifier, folder_path=load_folder)
        inverse_projection_NN.fit(X2d_train, Xnd_train, Y_train, 
                                  X2d_test, Xnd_test, Y_test, 
                                  epochs=epochs, batch_size=batch_size)
        return inverse_projection_NN
        
    def generate_boundary_map(self, 
                              Xnd_train: np.ndarray, Y_train: np.ndarray, 
                              Xnd_test: np.ndarray, Y_test: np.ndarray,
                              X2d_train: np.ndarray = None,
                              X2d_test: np.ndarray = None,
                              train_epochs:int=300, 
                              train_batch_size:int=32,
                              resolution:int=DBM_DEFAULT_RESOLUTION,
                              use_fast_decoding:bool=False,
                              projection:str='t-SNE'                              
                              ):
        """ Generates a 2D boundary map of the classifier's decision boundary.

        Args:
            X_train (np.ndarray): Training data set
            Y_train (np.ndarray): Training data labels
            X_test (np.ndarray): Testing data set
            Y_test (np.ndarray): Testing data labels
            X2d_train (np.ndarray | None): The 2D projection of the training data. If None, the data is projected using the given projection method. Defaults to None.
            X2d_test (np.ndarray | None): The 2D projection of the testing data. If None, the data is projected using the given projection method. Defaults to None.            
            train_epochs (int, optional): The number of epochs for which the DBM is trained. Defaults to 300.
            train_batch_size (int, optional): Train batch size. Defaults to 32.
            show_predictions (bool, optional): If set to true 10 prediction examples are shown. Defaults to True.
            resolution (int, optional): _description_. Defaults to DBM_DEFAULT_RESOLUTION.
            use_fast_decoding (bool, optional): If set to true the fast decoding method is used. Defaults to False.
            projection (str, optional): The projection method to be used. Defaults to 't-SNE'.
        
        Returns:
            img (np.array): A 2D numpy array with the decision boundary map, each element is an integer representing the class of the corresponding point.
            img_confidence (np.array): A 2D numpy array with the decision boundary map, each element is a float representing the confidence of the classifier for the corresponding point.
            X2d_train (np.array): A 2D matrix representing the projection of the training data set, each element is an integer representing the class of the corresponding point.
            X2d_test (np.array): A 2D matrix representing the projection of the testing data set, each element is an integer representing the class of the corresponding point.
            space_nd (np.array): A 2D matrix representing the nD space in which the decision boundary map is generated.
            history (dict): A dictionary containing the training history of the neural network.
        
        Example:
            >>> dbm = DBM(classifier)
            >>> img, img_confidence, X2d_train, X2d_test, space_nd, history = dbm.generate_boundary_map(X_train, Y_train, X_test, Y_test)
            >>> fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))
            >>> ax1.imshow(img)
            >>> ax2.imshow(img_confidence)
            >>> plt.show()
        """
        assert projection in ['t-SNE', 'PCA', 'UMAP']
        
        if X2d_train is None or X2d_test is None:
            Xnd_train_flatten = Xnd_train.reshape((Xnd_train.shape[0], -1))
            Xnd_test_flatten = Xnd_test.reshape((Xnd_test.shape[0], -1))
            X2d_train, X2d_test = self.__transform_2d__(Xnd_train_flatten, Xnd_test_flatten, projection)
        else:
            # Normalize the data to be in the range of [0,1]
            X2d_train, X2d_test = self.__normalize_2d__(X2d_train, X2d_test)
            
        if self.invNN is None:
            self.invNN = self.fit(X2d_train, Xnd_train, Y_train, 
                                  X2d_test, Xnd_test, Y_test, 
                                  train_epochs, train_batch_size,
                                  load_folder=os.path.join(DEFAULT_MODEL_PATH, projection))   
        
        self.console.log("Decoding the 2D space... 2D -> nD")
        
        save_img_path = os.path.join(DEFAULT_MODEL_PATH, "boundary_map")
        save_img_confidence_path = os.path.join(DEFAULT_MODEL_PATH, "boundary_map_confidence")
        
        if use_fast_decoding:
            img, img_confidence, spaceNd = self._get_img_dbm_fast_(resolution)
            save_img_path += "_fast"
            save_img_confidence_path += "_fast"
        else:
            img, img_confidence, spaceNd = self._get_img_dbm_(resolution)
    
        with open(f"{save_img_path}.npy", 'wb') as f:
            np.save(f, img)
        with open(f"{save_img_confidence_path}.npy", 'wb') as f:
            np.save(f, img_confidence)        

        self.resolution = resolution   
        self.spaceNd = spaceNd     
        self.X2d = np.concatenate((X2d_train, X2d_test), axis=0)
        self.Xnd = np.concatenate((Xnd_train.reshape((Xnd_train.shape[0],-1)), Xnd_test.reshape((Xnd_test.shape[0],-1))), axis=0)
        self.console.log("Map the 2D embedding of the data to the 2D image")
        
        # transform the encoded data to be in the range [0, resolution)
        X2d_train *= (resolution - 1)
        X2d_test *= (resolution - 1)
        X2d_train = X2d_train.astype(int)
        X2d_test = X2d_test.astype(int)
        
        for [i,j] in X2d_train:
            img[i,j] = -1
            img_confidence[i,j] = 1
        for [i,j] in X2d_test:
            img[i,j] = -2
            img_confidence[i,j] = 1
            
        
        with open(os.path.join(DEFAULT_MODEL_PATH, projection, "history.json"), 'r') as f:
            history = json.load(f)
        
        return (img, img_confidence, X2d_train, X2d_test, spaceNd, history)
    
    def generate_projection_errors(self, Xnd: np.ndarray = None, X2d: np.ndarray = None, resolution: int = None):
        """ Calculates the projection errors of the given data.

        Args:
            Xnd (np.array): The data to be projected. The data must be in range [0,1].
            X2d (np.array): The 2D projection of the data. The data must be in range [0,1].
            resolution (int): The resolution of the 2D space.
        Returns:
            errors (np.ndarray): The projection errors matrix of the given data. (resolution x resolution)
        """
        if resolution is None:
            if self.resolution is None:
                self.console.error("The resolution of the 2D space is not set, try to call the method 'generate_boundary_map' first.")
                raise Exception("The resolution of the 2D space is not set, try to call the method 'generate_boundary_map' first.")
            resolution = self.resolution
        if Xnd is None:
            if self.Xnd is None:
                self.console.error("The nD data is not set, try to call the method 'generate_boundary_map' first.")
                raise Exception("The nD data is not set, try to call the method 'generate_boundary_map' first.")
            Xnd = self.Xnd
        if X2d is None:
            if self.X2d is None:
                self.console.error("The 2D data is not set, try to call the method 'generate_boundary_map' first.")
                raise Exception("The 2D data is not set, try to call the method 'generate_boundary_map' first.")
            X2d = self.X2d
        
        assert len(Xnd) == len(X2d)
        
        X2d = X2d.reshape((X2d.shape[0], -1))
        Xnd = Xnd.reshape((Xnd.shape[0], -1))
        
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
     
    def _predict2dspace_(self, X2d: np.ndarray):
        """ Predicts the labels for the given 2D data set.

        Args:
            X2d (np.ndarray): The 2D data set
        
        Returns:
            predicted_labels (np.ndarray): The predicted labels for the given 2D data set
            predicted_confidence (np.ndarray): The predicted probabilities for the given 2D data set
            spaceNd (np.ndarray): The decoded nD space
        """
        spaceNd, predictions = self.invNN.decode(X2d, verbose=0)
        predicted_labels = np.array([np.argmax(p) for p in predictions])
        predicted_confidence = np.array([np.max(p) for p in predictions])
        return predicted_labels, predicted_confidence, spaceNd
 
    def __transform_2d__(self, X_train: np.ndarray, X_test: np.ndarray, projection:str='t-SNE'):
        """ Transforms the given data to 2D using a projection method.

        Args:
            X_train (np.ndarray): The training data.
            X_test (np.ndarray): The test data.
            projection (str, optional): The projection method to be used. Defaults to 't-SNE'.
            
        Returns:
            X2d (np.ndarray): The transformed data in 2D.
        """
        self.console.log(f"Transforming the data to 2D using {projection}")
        
        X = np.concatenate((X_train, X_test), axis=0)
        X2d = PROJECTION_METHODS[projection](X)
            
        self.console.log(f"Finished transforming the data to 2D using {projection}")
        X2d_train = X2d[:len(X_train)]
        X2d_test = X2d[len(X_train):]
        
        # rescale to [0,1]
        X2d_train, X2d_test = self.__normalize_2d__(X2d_train, X2d_test)
        # ---------------------
        if not os.path.exists(os.path.join(DEFAULT_MODEL_PATH, projection)):
            os.makedirs(os.path.join(DEFAULT_MODEL_PATH, projection))
            
        file_path = os.path.join(DEFAULT_MODEL_PATH, projection, "train_2d.npy")
        self.console.log("Saving the 2D data to the disk: " + file_path)
        with open(file_path, 'wb') as f:
            np.save(f, X2d_train)
        
        file_path = os.path.join(DEFAULT_MODEL_PATH, projection, "test_2d.npy")
        self.console.log("Saving the 2D data to the disk: " + file_path)
        with open(file_path, 'wb') as f:
            np.save(f, X2d_test)
        
        return X2d_train, X2d_test
    
    def __normalize_2d__(self, X2d_train: np.ndarray, X2d_test: np.ndarray):
        """ Normalizes the given 2D data to [0,1].

        Args:
            X2d_train (np.ndarray): training data
            X2d_test (np.ndarray): test data

        Returns:
            X2d_train (np.ndarray): normalized training data
            X2d_test (np.ndarray): normalized test data
        """
        x_min = min(np.min(X2d_train[:,0]), np.min(X2d_test[:,0]))
        y_min = min(np.min(X2d_train[:,1]), np.min(X2d_test[:,1]))
        x_max = max(np.max(X2d_train[:,0]), np.max(X2d_test[:,0]))
        y_max = max(np.max(X2d_train[:,1]), np.max(X2d_test[:,1]))
        X2d_train = (X2d_train - np.array([x_min, y_min])) / np.array([x_max - x_min, y_max - y_min])
        X2d_test = (X2d_test - np.array([x_min, y_min])) / np.array([x_max - x_min, y_max - y_min])
        return X2d_train, X2d_test