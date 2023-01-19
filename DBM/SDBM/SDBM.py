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
from DBM.SDBM.Autoencoder import DEFAULT_MODEL_PATH, Autoencoder, build_autoencoder
from DBM.tools import get_inv_proj_error, get_proj_error
from utils.tools import track_time_wrapper
        
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
            load_folder = DEFAULT_MODEL_PATH):
        
        # Train an autoencoder on the training data (this will be used to reduce the dimensionality of the data) nD -> 2D
        try:
            autoencoder = Autoencoder(folder_path = load_folder, load = True)
            self.console.log("Loaded autoencoder from disk")
        except Exception as e:
            self.console.log("Could not load autoencoder from disk. Training a new one.")
            data_shape = X_train.shape[1:]
            autoencoder = build_autoencoder(self.classifier, data_shape, show_summary=True)
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
                              use_fast_decoding=True, 
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
        self.console.log("Decoding the 2D space... 2D -> nD")
        
        img, img_confidence = self._get_img_dbm_fast_((min_x, max_x, min_y, max_y), resolution)
        
        with open(os.path.join(DEFAULT_MODEL_PATH, "fast_boundary_map.npy"), 'wb') as f:
            np.save(f, img)
        with open(os.path.join(DEFAULT_MODEL_PATH, "fast_boundary_map_confidence.npy"), 'wb') as f:
            np.save(f, img_confidence)
        
        img, img_confidence, space2d, spaceNd = self._get_img_dbm_((min_x, max_x, min_y, max_y), resolution)
        
        save_img_path = os.path.join(DEFAULT_MODEL_PATH, "boundary_map")
        save_img_confidence_path = os.path.join(DEFAULT_MODEL_PATH, "boundary_map_confidence")
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
        
        #img_projection_errors = self.get_projection_errors(spaceNd, space2d, img.flatten() , resolution)
        img_projection_errors = np.zeros((resolution, resolution))
        img_inverse_projection_errors = self.get_inverse_projection_errors(spaceNd.reshape((resolution, resolution, -1)))
      
        return (img, img_confidence, img_projection_errors, img_inverse_projection_errors, encoded_training_data, encoded_testing_data)
    
    def _predict2dspace_(self, X2d):
        """ Predicts the labels for the given 2D data set.

        Args:
            X2d (np.array): The 2D data set
        
        Returns:
            np.array: The predicted labels for the given 2D data set
            np.array: The predicted probabilities for the given 2D data set
            np.array: The decoded nD space
        """
        spaceNd = self.autoencoder.decode(X2d)
        predictions = self.classifier.predict(spaceNd, verbose=0)
        predicted_labels = np.array([np.argmax(p) for p in predictions])
        predicted_confidence = np.array([np.max(p) for p in predictions])
        return predicted_labels, predicted_confidence, spaceNd
    
        
    def get_projection_errors(self, Xnd, X2d, labels, resolution):
        """ Calculates the projection errors of the given data.

        Args:
            Xnd (np.array): The data to be projected.
            X2d (np.array): The 2D projection of the data.
        Returns:
            np.array: The projection errors matrix of the 
        """
        self.console.log("Calculating the projection errors of the given data")
        errors = np.zeros(X2d.shape[0])
        
        distances_2d = np.array([np.linalg.norm(x) for x in X2d])
        distances_nd = np.array([np.linalg.norm(x) for x in Xnd])
        
        indices_2d = np.argsort(distances_2d)
        indices_nd = np.argsort(distances_nd)
        
        for i in range(X2d.shape[0]):
            errors[i] = get_proj_error(i, indices_nd, indices_2d, labels)    
        
        # reshaping the errors to be in the shape of the 2d space
        errors = errors.reshape((resolution, resolution))
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