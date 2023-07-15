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

from .NNInv import DEFAULT_MODEL_PATH, NNInv
from .projections import PROJECTION_METHODS


from ..AbstractDBM import AbstractDBM, DBM_DEFAULT_RESOLUTION, DEFAULT_TRAINING_EPOCHS, DEFAULT_BATCH_SIZE, FAST_DBM_STRATEGIES

from ...utils import track_time_wrapper, TRAIN_DATA_POINT_MARKER, TEST_DATA_POINT_MARKER, TRAIN_2D_FILE_NAME, TEST_2D_FILE_NAME
from ...Logger import LoggerInterface, Logger
time_tracker_console = Logger(name="Decision Boundary Mapper - DBM", info_color="cyan", show_init=False)

CUSTOM_PROJECTION_NAME = "Custom"

class DBM(AbstractDBM):
    """
    This is a class that generated the DBM
    The core purpose of this class is to learn the inverse projection given a direct projection and construct the decision boundary map of a given classifier
    
    Public methods:
        fit: Learns an inverse projection by training a neural network. \n
        get_decision_boundary_map: Returns the decision boundary map for the given classifier. \n

    Example:
        >>> from DBM import DBM
        >>> import matplotlib.pyplot as plt
        >>> classifier = ...
        >>> dbm = DBM(classifier)
        >>> img, img_confidence, encoded_2d_train, encoded_2d_test = dbm.get_decision_boundary_map(Xnd_train, Xnd_test, X2d_train, X2d_test)
        >>> plt.imshow(img)
        >>> plt.show()
    """

    def __init__(self, classifier, logger: LoggerInterface | None = None):
        """ 
        Initializes the DBM class.

        Args:
            classifier (tensorflow.keras.Model): The classifier to be used for the decision boundary map.
            logger (LoggerInterface, optional): The logger for outputting info messages. Defaults to console logging.
        """
        super().__init__(classifier, logger)
        self.neural_network = None  # type: ignore

    @track_time_wrapper(logger=time_tracker_console)
    def fit(self,
            X2d: np.ndarray, Xnd: np.ndarray,
            epochs: int = DEFAULT_TRAINING_EPOCHS, batch_size: int = DEFAULT_BATCH_SIZE,
            load_folder: str = DEFAULT_MODEL_PATH):
        """ 
        Learns the inverse projection on the given data set.

        Args:
            X2d (np.ndarray): Training data set 2D data got from the projection of the original data (e.g. PCA, t-SNE, UMAP)
            Xnd (np.ndarray): Training data set nD data (e.g. MNIST, CIFAR10) (i.e. the original data)
            epochs (int, optional): The number of epochs for which the DBM is trained. Defaults to 300.
            batch_size (int, optional): Train batch size. Defaults to 32.

        Returns:
            inverse_porjection_NN (NNInv): The trained inverse projection neural network.
        """

        inverse_projection_NN = NNInv(folder_path=load_folder, logger=self.console)
        inverse_projection_NN.fit(X2d, Xnd,
                                  epochs=epochs,
                                  batch_size=batch_size)
        return inverse_projection_NN

    def generate_boundary_map(self,
                              Xnd_train: np.ndarray,
                              Xnd_test: np.ndarray,
                              X2d_train: np.ndarray | None = None,
                              X2d_test: np.ndarray | None = None,
                              nn_train_epochs: int = DEFAULT_TRAINING_EPOCHS,
                              nn_train_batch_size: int = DEFAULT_BATCH_SIZE,
                              resolution: int = DBM_DEFAULT_RESOLUTION,
                              fast_decoding_strategy: FAST_DBM_STRATEGIES = FAST_DBM_STRATEGIES.NONE,
                              load_folder: str = DEFAULT_MODEL_PATH,
                              projection: str = 't-SNE'):
        """ 
        Generates a 2D boundary map of the classifier's decision boundary.

        Args:
            X_train (np.ndarray): Training data set (nD)
            X_test (np.ndarray): Testing data set (nD)
            X2d_train (np.ndarray | None): The 2D projection of the training data. If None, the data is projected using the given projection method. Defaults to None.
            X2d_test (np.ndarray | None): The 2D projection of the testing data. If None, the data is projected using the given projection method. Defaults to None.            
            nn_train_epochs (int, optional): The number of epochs for which the DBM is trained. Defaults to 300.
            nn_train_batch_size (int, optional): Train batch size. Defaults to 32.
            resolution (int, optional): The desired resolution of the DBM. Defaults to DBM_DEFAULT_RESOLUTION.
            fast_decoding_strategy (FAST_DBM_STRATEGIES, optional): The strategy to use in generating the DBM. Defaults to FAST_DBM_STRATEGIES.NONE.
            load_folder (str, optional): The folder in which the model will be stored or if exists loaded from. Defaults to DEFAULT_MODEL_PATH
            projection (str, optional): The projection method to be used. Defaults to 't-SNE'.

        Returns:
            img (np.array): A 2D numpy array with the decision boundary map, each element is an integer representing the class of the corresponding point.
            img_confidence (np.array): A 2D numpy array with the decision boundary map, each element is a float representing the confidence of the classifier for the corresponding point.
            encoded_2d_train (np.array): An array representing the projection of the training data set, each element is a tuple representing the coordinates and the class of the corresponding point.
            encoded_2d_test (np.array): An array representing the projection of the testing data set, each element is an tuple representing the coordinates and the class of the corresponding point.
    
        Example:
            >>> dbm = DBM(classifier)
            >>> img, img_confidence, _, _ = dbm.generate_boundary_map(X_train, X_test)
            >>> fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))
            >>> ax1.imshow(img)
            >>> ax2.imshow(img_confidence)
            >>> plt.show()
        """
       
        # adding projection method to the end of the load_folder path
        if projection != load_folder.split(os.sep)[-1]:
            load_folder = os.path.join(load_folder, projection)
            
        if X2d_train is None or X2d_test is None:
            assert projection in PROJECTION_METHODS.keys()
            Xnd_train_flatten = Xnd_train.reshape((Xnd_train.shape[0], -1))
            Xnd_test_flatten = Xnd_test.reshape((Xnd_test.shape[0], -1))
            X2d_train, X2d_test = self.__transform_2d__(Xnd_train_flatten, Xnd_test_flatten, load_folder, projection)
        else:
            # Normalize the data to be in the range of [0,1]
            X2d_train, X2d_test = self.__normalize_2d__(X2d_train, X2d_test)
            

        # creating a folder for the model if not present
        if not os.path.exists(os.path.join(load_folder)):
            os.makedirs(os.path.join(load_folder))


        if self.neural_network is None:
            X2d = np.concatenate((X2d_train, X2d_test), axis=0)
            Xnd = np.concatenate((Xnd_train, Xnd_test), axis=0)
            self.neural_network = self.fit(X2d, Xnd,
                                           epochs = nn_train_epochs, 
                                           batch_size = nn_train_batch_size,
                                           load_folder=load_folder)

        self.resolution = resolution

        self.console.log("Decoding the 2D space... 2D -> nD")

        img, img_confidence = self.get_dbm(fast_decoding_strategy, resolution, load_folder)

        self.X2d = np.concatenate((X2d_train, X2d_test), axis=0)
        self.Xnd = np.concatenate((Xnd_train.reshape((Xnd_train.shape[0], -1)), Xnd_test.reshape((Xnd_test.shape[0], -1))), axis=0)
        self.console.log("Map the 2D embedding of the data to the 2D image")

        # transform the encoded data to be in the range [0, resolution)
        X2d_train *= (resolution - 1)
        X2d_test *= (resolution - 1)
        X2d_train = X2d_train.astype(int)
        X2d_test = X2d_test.astype(int)

        encoded_2d_train = np.zeros((len(X2d_train), 3))
        encoded_2d_test = np.zeros((len(X2d_test), 3))

        for k in range(len(X2d_train)):
            [i, j] = X2d_train[k]
            encoded_2d_train[k] = [i, j, img[i, j]]
        for k in range(len(X2d_test)):
            [i, j] = X2d_test[k]
            encoded_2d_test[k] = [i, j, img[i, j]]

        for [i, j] in X2d_test:
            img[i, j] = TEST_DATA_POINT_MARKER
            img_confidence[i, j] = 1
        for [i, j] in X2d_train:
            img[i, j] = TRAIN_DATA_POINT_MARKER
            img_confidence[i, j] = 1

        return (img, img_confidence, encoded_2d_train, encoded_2d_test)

    def _predict2dspace_(self, X2d: np.ndarray):
        """ 
        Predicts the labels for the given 2D data set.

        Args:
            X2d (np.ndarray): The 2D data set

        Returns:
            predicted_labels (np.ndarray): The predicted labels for the given 2D data set
            predicted_confidence (np.ndarray): The predicted probabilities for the given 2D data set
        """
        spaceNd = self.neural_network.decode(X2d, verbose=0)
        predictions = self.classifier.predict(spaceNd, verbose=0)                     # type: ignore
        predicted_labels = np.array([np.argmax(p) for p in predictions])
        predicted_confidence = np.array([np.max(p) for p in predictions])
        return predicted_labels, predicted_confidence, predictions

    def __transform_2d__(self, X_train: np.ndarray, X_test: np.ndarray, folder: str = DEFAULT_MODEL_PATH, projection: str = 't-SNE'):
        """ 
        Transforms the given data to 2D using a projection method.

        Args:
            X_train (np.ndarray): The training data.
            X_test (np.ndarray): The test data.
            folder (str, optional): The folder where the 2D data will be stored. Defaults to DEFAULT_MODEL_PATH.
            projection (str, optional): The projection method to be used. Defaults to 't-SNE'.

        Returns:
            X2d_train (np.ndarray): The transformed train data in 2D.
            X2d_test (np.ndarray): The transformed test data in 2D.
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
        if not os.path.exists(os.path.join(folder)):
            os.makedirs(os.path.join(folder))

        file_path = os.path.join(folder, TRAIN_2D_FILE_NAME)
        self.console.log("Saving the 2D data to the disk: " + file_path)
        with open(file_path, 'wb') as f:
            np.save(f, X2d_train)

        file_path = os.path.join(folder, TEST_2D_FILE_NAME)
        self.console.log("Saving the 2D data to the disk: " + file_path)
        with open(file_path, 'wb') as f:
            np.save(f, X2d_test)

        return X2d_train, X2d_test

    def __normalize_2d__(self, X2d_train: np.ndarray, X2d_test: np.ndarray):
        """ 
        Normalizes the given 2D data to [0,1].

        Args:
            X2d_train (np.ndarray): training data
            X2d_test (np.ndarray): test data

        Returns:
            X2d_train (np.ndarray): normalized training data
            X2d_test (np.ndarray): normalized test data
        """
        x_min = min(np.min(X2d_train[:, 0]), np.min(X2d_test[:, 0]))
        y_min = min(np.min(X2d_train[:, 1]), np.min(X2d_test[:, 1]))
        x_max = max(np.max(X2d_train[:, 0]), np.max(X2d_test[:, 0]))
        y_max = max(np.max(X2d_train[:, 1]), np.max(X2d_test[:, 1]))
        mins = np.array([x_min, y_min])
        ranges = np.array([x_max - x_min, y_max - y_min])

        X2d_train = (X2d_train - mins) / ranges  # type: ignore
        X2d_test = (X2d_test - mins) / ranges   # type: ignore
        return X2d_train, X2d_test
