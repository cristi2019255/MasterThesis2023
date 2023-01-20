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
from sklearn.neighbors import KDTree

from DBM.DBM.invNN import DEFAULT_MODEL_PATH, invNN
from DBM.DBM.projections import PROJECTION_METHODS
from DBM.DBMInterface import DBMInterface
from DBM.DBMInterface import DBM_DEFAULT_RESOLUTION
from DBM.tools import get_inv_proj_error, get_proj_error
from Logger import LoggerInterface

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
            logger (LoggerInterface, optional): The logger for outputting info messages. Defaults to None.
        """
        super().__init__(classifier, logger)
        self.invNN = None
        
        # creating a folder for the model
        if not os.path.exists(DEFAULT_MODEL_PATH):
           os.makedirs(DEFAULT_MODEL_PATH) 
                                                  
    def fit(self, 
            X2d_train: np.ndarray, Xnd_train: np.ndarray, Y_train: np.ndarray,
            X2d_test: np.ndarray, Xnd_test: np.ndarray, Y_test: np.ndarray, 
            epochs:int=10, batch_size:int=128, load_folder:str=DEFAULT_MODEL_PATH):
        """ 
        Trains the classifier on the given data set.

        Args:
            X2d_train (np.ndarray): Training data set 2D data got from the projection of the original data (e.g. PCA, t-SNE, UMAP)
            Xnd_train (np.ndarray): Training data set nD data (e.g. MNIST, CIFAR10) (i.e. the original data)
            Y_train (np.ndarray): Training data set labels
            X2d_test (np.ndarray): Testing data set 2D data got from the projection of the original data (e.g. PCA, t-SNE, UMAP)
            Xnd_test (np.ndarray): Testing data set nD data (e.g. MNIST, CIFAR10) (i.e. the original data)
            Y_test (np.ndarray): Testing data set nD data (e.g. MNIST, CIFAR10) (i.e. the original data)
            epochs (int, optional): The number of epochs for which the DBM is trained. Defaults to 10.
            batch_size (int, optional): Train batch size. Defaults to 128.
        
        Returns:
            inverse_porjection_NN (invNN): The trained inverse projection neural network.
        """
        
        inverse_projection_NN = invNN(classifier=self.classifier)
        inverse_projection_NN.fit(X2d_train, Xnd_train, Y_train, 
                                      X2d_test, Xnd_test, Y_test, 
                                      epochs=epochs, batch_size=batch_size)
            
        self.invNN = inverse_projection_NN
        self.classifier = self.invNN.classifier
        return inverse_projection_NN
        
    def generate_boundary_map(self, 
                              Xnd_train: np.ndarray, Y_train: np.ndarray, 
                              Xnd_test: np.ndarray, Y_test: np.ndarray,
                              X2d_train: np.ndarray = None,
                              X2d_test: np.ndarray = None,
                              train_epochs:int=10, 
                              train_batch_size:int=128,
                              resolution:int=DBM_DEFAULT_RESOLUTION,
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
            train_epochs (int, optional): The number of epochs for which the DBM is trained. Defaults to 10.
            train_batch_size (int, optional): Train batch size. Defaults to 128.
            show_predictions (bool, optional): If set to true 10 prediction examples are shown. Defaults to True.
            resolution (int, optional): _description_. Defaults to DBM_DEFAULT_RESOLUTION.
        
        Returns:
            img (np.array): A 2D numpy array with the decision boundary map, each element is an integer representing the class of the corresponding point.
            img_confidence (np.array): A 2D numpy array with the decision boundary map, each element is a float representing the confidence of the classifier for the corresponding point.
            img_projection_errors (np.array): A 2D numpy array with the decision boundary map, each element is a float representing the error of the projection for the corresponding point.
            img_inverse_projection_errors (np.array): A 2D numpy array with the decision boundary map, each element is a float representing the error of the inverse projection for the corresponding point.
            X2d_train (np.array): A 2D matrix representing the projection of the training data set, each element is an integer representing the class of the corresponding point.
            X2d_test (np.array): A 2D matrix representing the projection of the testing data set, each element is an integer representing the class of the corresponding point.
        
        Example:
            >>> dbm = DBM(classifier)
            >>> img, img_confidence, img_projection_errors, img_inverse_projection_errors, X2d_train, X2d_test = dbm.generate_boundary_map(X_train, Y_train, X_test, Y_test)
            >>> plt.imshow(img)
            >>> plt.show()
        """
        assert projection in ['t-SNE', 'PCA', 'UMAP']
        
        if X2d_train is None:
            X2d_train = self.__transform_2d__(Xnd_train.reshape((Xnd_train.shape[0], -1)), projection, data_name = "train")
            
        if X2d_test is None:
            X2d_test = self.__transform_2d__(Xnd_test.reshape((Xnd_test.shape[0], -1)), projection, data_name = "test")

        if self.invNN is None:
            self.fit(X2d_train, Xnd_train, Y_train, 
                     X2d_test, Xnd_test, Y_test, 
                     train_epochs, train_batch_size)   
        
        min_x = min(np.min(X2d_train[:,0]), np.min(X2d_test[:,0]))
        max_x = max(np.max(X2d_train[:,0]), np.max(X2d_test[:,0]))
        min_y = min(np.min(X2d_train[:,1]), np.min(X2d_test[:,1]))
        max_y = max(np.max(X2d_train[:,1]), np.max(X2d_test[:,1]))
        
        self.console.log("Decoding the 2D space... 2D -> nD")
        
        """    
        img, img_confidence = self._get_img_dbm_fast_((min_x, max_x, min_y, max_y), resolution)
        save_img_path = os.path.join(DEFAULT_MODEL_PATH, "fast_boundary_map")
        save_img_confidence_path = os.path.join(DEFAULT_MODEL_PATH, "fast_boundary_map_confidence")
        with open(f"{save_img_path}.npy", 'wb') as f:
            np.save(f, img)
        with open(f"{save_img_confidence_path}.npy", 'wb') as f:
            np.save(f, img_confidence)        
        """
        
        img, img_confidence, space2d, spaceNd = self._get_img_dbm_((min_x, max_x, min_y, max_y), resolution)
        
        save_img_path = os.path.join(DEFAULT_MODEL_PATH, "boundary_map")
        save_img_confidence_path = os.path.join(DEFAULT_MODEL_PATH, "boundary_map_confidence")
        with open(f"{save_img_path}.npy", 'wb') as f:
            np.save(f, img)
        with open(f"{save_img_confidence_path}.npy", 'wb') as f:
            np.save(f, img_confidence)        

        X2d = np.concatenate((X2d_train, X2d_test), axis=0)
        Xnd = np.concatenate((Xnd_train.reshape((Xnd_train.shape[0],-1)), Xnd_test.reshape((Xnd_test.shape[0],-1))), axis=0)
        self.console.log("Map the 2D embedding of the data to the 2D image")
        # transform the encoded data to be in the range [0, resolution]
        X2d_train[:,0] = (X2d_train[:,0] - min_x) / (max_x - min_x) * (resolution - 1)
        X2d_train[:,1] = (X2d_train[:,1] - min_y) / (max_y - min_y) * (resolution - 1)
        X2d_test[:,0] = (X2d_test[:,0] - min_x) / (max_x - min_x) * (resolution - 1)
        X2d_test[:,1] = (X2d_test[:,1] - min_y) / (max_y - min_y) * (resolution - 1)
        
        X2d_train = X2d_train.astype(int)
        X2d_test = X2d_test.astype(int)
        
        for [i,j] in X2d_train:
            img[i,j] = -1
            img_confidence[i,j] = 1
        for [i,j] in X2d_test:
            img[i,j] = -2
            img_confidence[i,j] = 1
        
        img_projection_errors = self.get_projection_errors((min_x, max_x, min_y, max_y), Xnd, X2d, resolution) 
        
        img_inverse_projection_errors = self.get_inverse_projection_errors(spaceNd.reshape((resolution, resolution, -1)))
        
        return (img, img_confidence, img_projection_errors, img_inverse_projection_errors, X2d_train, X2d_test)
    
    def _predict2dspace_(self, X2d: np.ndarray):
        """ Predicts the labels for the given 2D data set.

        Args:
            X2d (np.ndarray): The 2D data set
        
        Returns:
            predicted_labels (np.ndarray): The predicted labels for the given 2D data set
            predicted_confidence (np.ndarray): The predicted probabilities for the given 2D data set
            spaceNd (np.ndarray): The decoded nD space
        """
        spaceNd = self.invNN.decode(X2d)
        predictions = self.invNN.classifier.predict(spaceNd, verbose=0)
        predicted_labels = np.array([np.argmax(p) for p in predictions])
        predicted_confidence = np.array([np.max(p) for p in predictions])
        return predicted_labels, predicted_confidence, spaceNd
    
    def get_projection_errors(self, boundaries: tuple, Xnd: np.ndarray, X2d: np.ndarray, resolution: int):
        """ Calculates the projection errors of the given data.

        Args:
            boundaries ((float,float,float,float)): The boundaries of the 2D space. (min_x, max_x, min_y, max_y)
            Xnd (np.array): The data to be projected.
            X2d (np.array): The 2D projection of the data.
            resolution (int): The resolution of the 2D space.
        Returns:
            errors (np.ndarray): The projection errors matrix of the given data. (resolution x resolution)
        """
        self.console.log("Calculating the projection errors of the given data")
        errors = np.zeros((resolution,resolution))
        min_x, max_x, min_y, max_y = boundaries
        
        K = 10
        metric = "euclidean"
        tree = KDTree(X2d, metric=metric)
        indices_embedded = tree.query(X2d, k=K, return_distance=False)
        # Drop the actual point itself
        indices_embedded = indices_embedded[:, 1:]
        self.console.log("Finished computing the 2D tree")
        
        tree = KDTree(Xnd, metric=metric)
        indices_source = tree.query(Xnd, k=K, return_distance=False)
        # Drop the actual point itself
        indices_source = indices_source[:, 1:]
        self.console.log("Finished computing the nD tree")
        
        for k in range(X2d.shape[0]):
            x, y = X2d[k]
            i, j = int((x - min_x) / (max_x - min_x) * (resolution - 1)), int((y - min_y) / (max_y - min_y) * (resolution - 1)) 
            errors[i,j] = get_proj_error(indices_source[k], indices_embedded[k])
        
        return errors
    
    def get_inverse_projection_errors(self, Xnd: np.ndarray):
        """ Calculates the inverse projection errors of the given data.

        Args:
            Xnd (np.array): The nd inverse projection of the data.
        
        Returns:
            errors (np.ndarray): The inverse projection errors matrix of the given data. (resolution x resolution)
        """
        self.console.log("Calculating the inverse projection errors of the given data")
        errors = np.zeros(Xnd.shape[:2])
        for i in range(Xnd.shape[0]):
            for j in range(Xnd.shape[1]):
                errors[i,j] = get_inv_proj_error(i,j, Xnd)
                
        # normalizing the errors to be in the range [0,1]
        errors = (errors - np.min(errors)) / (np.max(errors) - np.min(errors))
        return errors
        
    def __transform_2d__(self, X: np.ndarray, projection:str='t-SNE', data_name:str="training"):
        """ Transforms the given data to 2D using a projection method.

        Args:
            X (np.ndarray): The data to be transformed
            projection (str, optional): The projection method to be used. Defaults to 't-SNE'.
            data_name (str, optional): The name of the data. Defaults to "training".
        Returns:
            X2d (np.ndarray): The transformed data in 2D.
        """
        self.console.log(f"Transforming the data to 2D using {projection}")
        
        X2d = PROJECTION_METHODS[projection](X)
            
        self.console.log("Finished transforming the data to 2D using {projection}")
        
        file_path = os.path.join(DEFAULT_MODEL_PATH, projection + data_name + "_2d.npy")
        self.console.log("Saving the 2D data to the disk: " + file_path)
        with open(file_path, 'wb') as f:
            np.save(f, X2d)
        
        return X2d
    