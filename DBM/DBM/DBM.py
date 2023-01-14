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
from DBM.DBM.invNN import DEFAULT_MODEL_PATH, invNN
from DBM.DBMInterface import DBMInterface
from DBM.DBMInterface import DBM_DEFAULT_RESOLUTION
from DBM.DBM.projections import PROJECTION_METHODS
import numpy as np

from DBM.tools import get_inv_proj_error, get_proj_error


class DBM(DBMInterface):
    
    def __init__(self, classifier, logger=None):
        super().__init__(classifier, logger)
        self.invNN = None
        
        # creating a folder for the model
        if not os.path.exists(DEFAULT_MODEL_PATH):
           os.makedirs(DEFAULT_MODEL_PATH) 
                                                  
    def fit(self, 
            X2d_train, Xnd_train, Y_train,
            X2d_test, Xnd_test, Y_test, 
            epochs=10, batch_size=128, load_folder=DEFAULT_MODEL_PATH):
        """ Trains the classifier on the given data set.

        Args:
            X2d_train (np.array): Training data set 2D data got from the projection of the original data (e.g. PCA, t-SNE, UMAP)
            Xnd_train (np.array): Training data set nD data (e.g. MNIST, CIFAR10) (i.e. the original data)
            Y_train (np.array): Training data set labels
            X2d_test (np.array): Testing data set 2D data got from the projection of the original data (e.g. PCA, t-SNE, UMAP)
            Xnd_test (np.array): Testing data set nD data (e.g. MNIST, CIFAR10) (i.e. the original data)
            Y_test (np.array): Testing data set nD data (e.g. MNIST, CIFAR10) (i.e. the original data)
            epochs (int, optional): The number of epochs for which the DBM is trained. Defaults to 10.
            batch_size (int, optional): Train batch size. Defaults to 128.
        """
        
        inverse_porjection_NN = invNN(classifier=self.classifier)
        inverse_porjection_NN.fit(X2d_train, Xnd_train, Y_train, 
                                      X2d_test, Xnd_test, Y_test, 
                                      epochs=epochs, batch_size=batch_size)
            
        self.invNN = inverse_porjection_NN
        self.classifier = self.invNN.classifier
        return inverse_porjection_NN
        
    def generate_boundary_map(self, 
                              Xnd_train, Y_train, 
                              Xnd_test, Y_test,
                              X2d_train = None,
                              X2d_test = None,
                              train_epochs=10, 
                              train_batch_size=128,
                              resolution=DBM_DEFAULT_RESOLUTION,
                              projection='t-SNE'                              
                              ):
        """ Generates a 2D boundary map of the classifier's decision boundary.

        Args:
            X_train (np.array): Training data set
            Y_train (np.array): Training data labels
            X_test (np.array): Testing data set
            Y_test (np.array): Testing data labels
            train_epochs (int, optional): The number of epochs for which the DBM is trained. Defaults to 10.
            train_batch_size (int, optional): Train batch size. Defaults to 128.
            show_predictions (bool, optional): If set to true 10 prediction examples are shown. Defaults to True.
            resolution (_type_, optional): _description_. Defaults to DBM_DEFAULT_RESOLUTION.
        
        Returns:
            img (np.array): A 2D numpy array with the decision boundary map, each element is an integer representing the class of the corresponding point.
            img_confidence (np.array): A 2D numpy array with the decision boundary map, each element is a float representing the confidence of the classifier for the corresponding point.
            img_projection_errors (np.array): A 2D numpy array with the decision boundary map, each element is a float representing the error of the projection for the corresponding point.
            img_inverse_projection_errors (np.array): A 2D numpy array with the decision boundary map, each element is a float representing the error of the inverse projection for the corresponding point.
            X2d_train (np.array): A 2D matrix representing the projection of the training data set, each element is an integer representing the class of the corresponding point.
            X2d_test (np.array): A 2D matrix representing the projection of the testing data set, each element is an integer representing the class of the corresponding point.
        """
        assert type(Xnd_train) == np.ndarray
        assert type(Y_train) == np.ndarray
        assert type(Xnd_test) == np.ndarray
        assert type(Y_test) == np.ndarray
        assert projection in ['t-SNE', 'PCA', 'UMAP']
        
        if X2d_train is None:
            X2d_train = self.transform_2d(Xnd_train, projection, data_name = "train")
            
        if X2d_test is None:
            X2d_test = self.transform_2d(Xnd_test, projection, data_name = "test")

        if self.invNN is None:
            self.fit(X2d_train, Xnd_train, Y_train, 
                     X2d_test, Xnd_test, Y_test, 
                     train_epochs, train_batch_size)   
        
        min_x = min(np.min(X2d_train[:,0]), np.min(X2d_test[:,0]))
        max_x = max(np.max(X2d_train[:,0]), np.max(X2d_test[:,0]))
        min_y = min(np.min(X2d_train[:,1]), np.min(X2d_test[:,1]))
        max_y = max(np.max(X2d_train[:,1]), np.max(X2d_test[:,1]))
            
        space2d = np.array([(i / resolution * (max_x - min_x) + min_x, j / resolution * (max_y - min_y) + min_y) for i in range(resolution) for j in range(resolution)])
        
        self.console.log("Decoding the 2D space... 2D -> nD -> 1D")
        spaceNd = self.invNN.decoder.predict(space2d)
        predictions = self.invNN.classifier.predict(spaceNd)
        predicted_labels = np.array([np.argmax(p) for p in predictions])
        predicted_confidence = np.array([np.max(p) for p in predictions])
        img = predicted_labels.reshape((resolution, resolution))
        img_confidence = predicted_confidence.reshape((resolution, resolution))
        
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
            
        save_img_path = os.path.join(DEFAULT_MODEL_PATH, "boundary_map")
        save_img_confidence_path = os.path.join(DEFAULT_MODEL_PATH, "boundary_map_confidence")
        with open(f"{save_img_path}.npy", 'wb') as f:
            np.save(f, img)
        with open(f"{save_img_confidence_path}.npy", 'wb') as f:
            np.save(f, img_confidence)
        
        # to do, change space2d to X2d generated by the encoder projection
        img_projection_errors = self.get_projection_errors(spaceNd.reshape((resolution, resolution, -1)), space2d.reshape((resolution, resolution, -1))) 
        
        img_inverse_projection_errors = self.get_inverse_projection_errors(spaceNd.reshape((resolution, resolution, -1)))
        
        return (img, img_confidence, img_projection_errors, img_inverse_projection_errors, X2d_train, X2d_test)
    
    
    def get_projection_errors(self, Xnd, X2d):
        """ Calculates the projection errors of the given data.

        Args:
            Xnd (np.array): The data to be projected.
            X2d (np.array): The 2D projection of the data.
        Returns:
            np.array: The projection errors matrix of the 
        """
        self.console.log("Calculating the projection errors of the given data")
        errors = np.zeros(X2d.shape[:2])
        for i in range(X2d.shape[0]):
            for j in range(X2d.shape[1]):
                errors[i,j] = get_proj_error(i,j, Xnd, X2d)    
        
        # normalizing the errors to be in the range [0,1]
        errors = (errors - np.min(errors)) / (np.max(errors) - np.min(errors))
        return errors
    
    def get_inverse_projection_errors(self, Xnd):
        """ Calculates the inverse projection errors of the given data.

        Args:
            X2d (np.array): The 2d projection of the data.
            Xnd (np.array): The nd inverse projection of the data.
        """
        self.console.log("Calculating the inverse projection errors of the given data")
        errors = np.zeros(Xnd.shape[:2])
        for i in range(Xnd.shape[0]):
            for j in range(Xnd.shape[1]):
                errors[i,j] = get_inv_proj_error(i,j, Xnd)
                
        # normalizing the errors to be in the range [0,1]
        errors = (errors - np.min(errors)) / (np.max(errors) - np.min(errors))
        return errors
        
    def transform_2d(self, X, projection='t-SNE', data_name="training"):
        """ Transforms the given data to 2D using a projection method.

        Args:
            X (np.array): The data to be transformed

        Returns:
            np.array: The transformed data
        """
        self.console.log("Transforming the data to 2D using {projection}")
        
        X2d = PROJECTION_METHODS[projection](X)
            
        self.console.log("Finished transforming the data to 2D using {projection}")
        
        file_path = os.path.join(DEFAULT_MODEL_PATH, data_name + "_2d.npy")
        self.console.log("Saving the 2D data to the disk: " + file_path)
        with open(file_path, 'wb') as f:
            np.save(f, X2d)
        
        return X2d